"""Streamlit demo. Run:  streamlit run app.py   (Ollama must be running).
Do not run it while an experiment is running: both compete for the CPU and distort latency."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from src.agents.claim_extractor import ClaimExtractor
from src.agents.claim_verifier import ClaimVerifier, ClaimVerifierConfig
from src.agents.evidence_verifier import EvidenceVerifier
from src.agents.nli import NLIScorer
from src.agents.planner import QueryPlanner
from src.agents.retriever_agent import RetrievalAgent
from src.agents.risk_controller import RiskConfig, RiskController
from src.agents.web_retriever import WebRetrievalAgent
from src.config import DATA_PROCESSED, INDEX_DIR, RESULTS_DIR, ROOT, Config
from src.graph.state import make_condition
from src.graph.workflow import Pipeline
from src.llm.cache import CachedLLM
from src.llm.provider import get_llm
from src.retrieval.embeddings import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.web import CachingClient, WebClient
from src.utils import read_json

st.set_page_config(page_title="Self-Verifying RAG", layout="wide")
COND_NAMES = {"A": "A Vanilla RAG", "B": "B +Evidence verification", "C": "C +Claim verification",
              "D": "D +Evidence+Claim", "E": "E Risk-adaptive", "F": "F Always-on"}


@st.cache_resource(show_spinner="Loading models (the first run is slow)...")
def load_core():
    cfg = Config.from_env()
    embedder = Embedder(cfg.embedding_model, cfg.query_instruction)
    idx = INDEX_DIR / "hotpot_dev200"
    store = VectorStore.load(idx, embedder) if (idx / "index.faiss").exists() else None
    llm = CachedLLM(get_llm(cfg), ROOT / "evaluation" / "cache" / "llm_cache.jsonl")
    rp, tp = RESULTS_DIR / "risk_config.json", RESULTS_DIR / "claim_threshold.json"
    qp = DATA_PROCESSED / "questions.json"
    return {"cfg": cfg, "embedder": embedder, "store": store, "llm": llm, "nli": NLIScorer(),
            "risk_cfg": RiskConfig.load(rp) if rp.exists() else RiskConfig(),
            "support": float(read_json(tp)["support_threshold"]) if tp.exists() else ClaimVerifierConfig().support_threshold,
            "questions": read_json(qp) if qp.exists() else [],
            "calibrated": rp.exists()}


def run_condition(cond, question, retriever, core, policy):
    pipe = Pipeline(make_condition(cond, final_policy=policy), retriever, EvidenceVerifier(None),
                    RiskController(core["risk_cfg"]), core["llm"], ClaimExtractor(),
                    ClaimVerifier(core["nli"], ClaimVerifierConfig(support_threshold=core["support"])))
    llm = core["llm"]
    before, wall0, t0 = llm.usage.snapshot(), llm.wall_s, time.perf_counter()
    state = pipe.run(question)
    wall = time.perf_counter() - t0
    delta = llm.usage.delta(before)
    standalone = (wall - (llm.wall_s - wall0)) + delta["llm_latency_s"]
    return state, delta, standalone


def render(state, delta, standalone):
    if state["abstained"]:
        st.warning(f"Abstained -> {state['final_answer']}")
    else:
        st.success(state["final_answer"])
    if state.get("flagged_unverified"):
        st.info("Returned, but flagged: its claims could not be verified against the evidence.")
    if state.get("answer") and state["answer"] != state["final_answer"]:
        st.caption(f"Answer before the critic's decision: {state['answer']}")
    if state.get("statement"):
        st.caption(f"Generated statement (the claim that is verified): {state['statement']}")
    c = st.columns(5)
    c[0].metric("Risk level", state.get("risk_level", "NA"))
    c[1].metric("LLM calls", delta["llm_calls"])
    c[2].metric("Total tokens", delta["total_tokens"])
    c[3].metric("Standalone latency (s)", f"{standalone:.1f}")
    c[4].metric("Re-retrievals", state["round"])

    with st.expander("Source passages (final evidence)", expanded=True):
        for r in state["results"]:
            ch = r.chunk
            url = ch.metadata.get("url")
            head = f"**[{r.rank}] {ch.title or ch.doc_id}** - score {r.score:.3f}"
            st.markdown(head + (f" - [{url}]({url})" if url else ""))
            st.caption(ch.text[:700] + ("..." if len(ch.text) > 700 else ""))
    if state.get("evidence"):
        with st.expander("Evidence signals and risk"):
            ev = state["evidence"]
            st.dataframe(pd.DataFrame([{k: round(ev[k], 3) for k in
                         ("top1_score", "score_gap", "mean_score", "coverage", "retrieval_risk", "coverage_risk")}
                         | {"risk_score": round(state["risk_score"], 3), "risk_level": state["risk_level"]}]))
    if state.get("claim_results"):
        with st.expander("Claim verification", expanded=True):
            st.dataframe(pd.DataFrame([{"claim": x["claim"], "verdict": x["verdict"],
                                        "entailment": round(x["entailment"], 3),
                                        "contradiction": round(x["contradiction"], 3),
                                        "uncertain": x["uncertain"]} for x in state["claim_results"]]))
            if state.get("critic_action"):
                st.write(f"Critic's last decision: **{state['critic_action']}**")
    elif state.get("claim_verified") is False and state.get("risk_level") == "LOW":
        st.caption("Claim verification skipped (LOW risk, adaptive condition).")
    with st.expander("Workflow trace / revision history"):
        st.json(state["history"], expanded=False)


st.title("Risk-Adaptive Self-Verifying RAG")
core = load_core()
cfg = core["cfg"]

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Evidence source", ["Local corpus (HotpotQA subset)", "Live web"])
    conds = st.multiselect("Conditions to run/compare", list(COND_NAMES), default=["A", "E"],
                           format_func=COND_NAMES.get)
    policy = st.selectbox("When claims stay unsupported", ["abstain", "abstain_if_contradicted", "accept_flagged"])
    if mode == "Live web":
        backend = st.selectbox("Search backend", ["ddg", "wikipedia"])
        n_pages = st.slider("Pages per search", 2, 6, 3)
    st.caption(f"LLM: {cfg.llm_model} (local, Ollama)  |  embeddings: {cfg.embedding_model}")
    st.caption(f"risk thresholds: low {core['risk_cfg'].low_threshold:.2f} / high {core['risk_cfg'].high_threshold:.2f}"
               f"{'' if core['calibrated'] else ' (defaults, NOT calibrated)'}  |  claim threshold {core['support']:.2f}")
    st.caption("Each LLM call takes tens of seconds on CPU; the first call also loads the model.")

gold = None
if mode.startswith("Local"):
    if core["store"] is None:
        st.error("Local index not found. Run scripts/build_index.py first.")
        st.stop()
    opts = ["(type your own)"] + [q["question"] for q in core["questions"][:40]]
    pick = st.selectbox("Dataset question (test set)", opts)
    question = st.text_input("Question", value="" if pick == opts[0] else pick)
    gold = next((q for q in core["questions"] if q["question"] == question), None)
else:
    st.warning("Live web mode: risk thresholds were calibrated on HotpotQA and do not transfer to web content "
               "(risk labels are uncalibrated here). Web pages are untrusted text and may contain misleading "
               "or injected instructions; NLI verification does not protect against that.")
    question = st.text_input("Question about current or public information")

if st.button("Run", type="primary") and question.strip() and conds:
    if mode == "Live web":
        key = ("web_client", backend)
        if st.session_state.get("web_key") != key:
            st.session_state["web_client"] = CachingClient(WebClient(backend))
            st.session_state["web_key"] = key
        client = st.session_state["web_client"]
        if not client.search(question, n_pages * 2):
            st.error(f"No web results. {client.last_error or ''} Try the 'wikipedia' backend.")
            st.stop()

    outputs = {}
    for cond in conds:
        if mode.startswith("Local"):
            retriever = RetrievalAgent(core["store"], QueryPlanner(), top_k=cfg.top_k, extra=3)
        else:
            retriever = WebRetrievalAgent(core["embedder"], client, QueryPlanner(), top_k=cfg.top_k,
                                          n_pages=n_pages)
        try:
            with st.spinner(f"Running condition {cond} ..."):
                outputs[cond] = run_condition(cond, question, retriever, core, policy)
        except RuntimeError as exc:
            st.error(str(exc))
            break
    if outputs:
        if gold:
            st.info(f"Dataset gold answer: **{gold['answer']}**  (gold paragraphs: {len(gold['gold_doc_ids'])})")
        st.subheader("Comparison")
        st.dataframe(pd.DataFrame([{
            "condition": COND_NAMES[c], "answer": s["final_answer"], "abstained": s["abstained"],
            "risk": s.get("risk_level"), "claims_verified": s.get("claim_verified"), "re_retrievals": s["round"],
            "LLM calls": d["llm_calls"], "tokens": d["total_tokens"], "latency_s": round(lat, 1)}
            for c, (s, d, lat) in outputs.items()]))
        tabs = st.tabs([COND_NAMES[c] for c in outputs])
        for tab, (c, (s, d, lat)) in zip(tabs, outputs.items()):
            with tab:
                render(s, d, lat)
