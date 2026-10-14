"""Mac CPU에서 스케줄러 흐름을 디버거로 따라가기 위한 최소 실행 스크립트.

VLLM_ENABLE_V1_MULTIPROCESSING=0 으로 실행해야 EngineCore가 같은 프로세스에서 돌아
vllm/v1/core/sched/scheduler.py 의 브레이크포인트가 걸린다.
"""

from vllm import LLM, SamplingParams

prompts = [
    "Continuous batching in one sentence:",
    "The capital of Korea is",
    "Write a haiku about GPUs:",
]

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    dtype="float16",
    max_model_len=1024,
    max_num_batched_tokens=64,  # 작게 잡아서 chunked prefill이 일어나게 함
    max_num_seqs=4,
)
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))

for out in outputs:
    print(repr(out.prompt), "->", repr(out.outputs[0].text))
