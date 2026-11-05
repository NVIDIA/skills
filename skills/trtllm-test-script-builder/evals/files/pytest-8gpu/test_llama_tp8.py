# Synthetic excerpt of an integration test, used as an evaluation fixture.
import pytest


@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("backend", ["pytorch"])
def test_llama_70b_tp8(backend, llm_models_root):
    """Llama-3.1-70B generation smoke test at TP8."""
    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(model=f"{llm_models_root}/llama-3.1-model/Llama-3.1-70B-Instruct",
              tensor_parallel_size=8,
              backend=backend)
    outputs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8))
    assert "Paris" in outputs[0].outputs[0].text
