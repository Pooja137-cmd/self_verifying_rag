"""Build corpus.jsonl + questions.json from HotpotQA (distractor dev set).

All paragraphs of the sampled questions are pooled into ONE corpus, so retrieval
must find gold paragraphs among everyone's distractors (harder and more
realistic than the original per-question 10-paragraph setting).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, DATA_RAW  # noqa: E402
from src.utils import read_json, sha256_file, write_json, write_jsonl  # noqa: E402

URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def make_doc_id(title: str, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "doc"
    return f"{slug}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DATA_RAW / "hotpot_dev_distractor_v1.json"))
    ap.add_argument("--n", type=int, default=200, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DATA_PROCESSED))
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        print(f"Downloading {URL} ...")
        try:
            download(URL, raw)
        except Exception as exc:
            print(f"Download failed: {exc}\nDownload hotpot_dev_distractor_v1.json manually from "
                  f"https://hotpotqa.github.io/ and save it to:\n  {raw}\nthen rerun.")
            sys.exit(1)

    data = read_json(raw)
    rng = random.Random(args.seed)
    n = min(args.n, len(data))
    picked = sorted(rng.sample(range(len(data)), n))

    docs: dict[str, dict] = {}
    questions: list[dict] = []
    missing_gold = 0
    for i in picked:
        ex = data[i]
        title_to_id: dict[str, str] = {}
        for title, sents in ex["context"]:
            text = " ".join(s.strip() for s in sents).strip()
            did = make_doc_id(title, text)
            docs.setdefault(did, {"doc_id": did, "title": title, "text": text, "metadata": {"source": "hotpotqa"}})
            title_to_id[title] = did
        gold_titles = {t for t, _ in ex["supporting_facts"]}
        gold_ids = sorted(title_to_id[t] for t in gold_titles if t in title_to_id)
        missing_gold += len(gold_titles) - len(gold_ids)
        questions.append({
            "id": ex["_id"], "question": ex["question"], "answer": ex["answer"],
            "type": ex.get("type"), "level": ex.get("level"),
            "gold_doc_ids": gold_ids, "context_doc_ids": sorted(set(title_to_id.values())),
        })

    out = Path(args.out)
    n_docs = write_jsonl(out / "corpus.jsonl", docs.values())
    write_json(out / "questions.json", questions)
    write_json(out / "manifest.json", {
        "source": "HotpotQA dev distractor", "source_url": URL, "raw_sha256": sha256_file(raw),
        "n_questions": len(questions), "n_docs": n_docs, "seed": args.seed,
        "gold_titles_missing_from_context": missing_gold,
    })
    print(f"questions: {len(questions)}   unique paragraphs (docs): {n_docs}")
    print(f"gold titles missing from context: {missing_gold}")
    print(f"wrote {out / 'corpus.jsonl'}, {out / 'questions.json'}, {out / 'manifest.json'}")


if __name__ == "__main__":
    main()