from src.llm.base import LLMClient, LLMResponse
from src.pipeline.vanilla import clean_answer, run_vanilla_rag
from tests.test_vector_store import build


class FakeLLM(LLMClient):
    model = "fake"

    def _generate(self, prompt, system, json_mode, max_tokens):
        assert "Question:" in prompt and "Context:" in prompt
        return LLMResponse("Answer: Paris\nextra line", 100, 5, 0.01, "fake")


def test_clean_answer():
    assert clean_answer("Answer: Paris\nblah") == "Paris"
    assert clean_answer("") == ""


def test_run_vanilla_rag_logs_everything():
    llm = FakeLLM()
    rec = run_vanilla_rag("Where is the Eiffel Tower?", build(), llm, top_k=2)
    assert rec["answer"] == "Paris"
    assert len(rec["retrieved"]) == 2 and rec["retrieved"][0]["doc_id"] == "eiffel"
    assert rec["llm_calls"] == 1 and rec["total_tokens"] == 105
    assert llm.usage.calls == 1