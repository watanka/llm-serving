### vllm 관련 이해한 내용 소화

source: https://vllm.ai/blog/2025-09-05-anatomy-of-vllm

1. LLM engine & engine core
2. Advanced features: chunked prefill, prefix caching, guided & speculative decoding, disaggregated P/D
3. Scaling up: from single-GPU to multi-GPU execution
4. Serving layer: distributed / concurrent web scaffolding
5. benchmarks and auto-tuning




### LLM engine & engine core

단순한 모델부터 시작. 
standard transformer에 synchronous, one single gpu, 그리고 No advanced features like kv-cache, ...

LLM Engine Constructor
구성 요소
- vllm config
- processor(raw inputs -> EngineCoreRequests via validation, tokenization, and processing)
- engine core client (from `InprocClient` -> `EngineCore`)
- output processor (EngineCoreOutputs -> RequestOutput)
- Engine core는 몇개의 subcomponents로 이뤄져있음
  - model executor
  - structured output manager
  - scheduler: contains policy settings, queues, kv-cache manager

kv-cache에 사용할 block pool에서 index들 관리



Model Executor construction 동안
1. init device
2. load model
3. initialize kv-cache

가 일어남.


