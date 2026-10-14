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

#### LLM Engine Constructor
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



#### Generate Function
preprocessor -> enginecorerequest -> engine core -> Request -> added to scheduler as `waiting`
engine repeatedly call `step()` function
each step function has three stages
1. schedule
2. forward pass
3. postprocess


Scheduler
- v1에서는 prefill/decode 상관없이 한번에 처리 가능
- length 상관없이 1차원으로 처리.
- decode 먼저 받고, prefill.
- decode/prefill 둘다
  - 1. 생성할 new token 수 계산
    2. kv cache manager의 `allocate_slots` 함수 호출
    3. token budget 업데이트
 
allocate_slots가 하는일
1. kv cache blocks가 몇개 필요한지 계산
2. 풀에 충분한 블락이 있는지 확인. 부족하다면 선점 조건 계산해서 low-priority request evict하기도함
3. 블락 할당

kv-cache 블락이 정확히 어떤 개념인지 좀 이해를 해야할듯.
여기 안에는 텐서들이 들어간건가
블락을 나누는 기준은?


forward pass
이제 execute_model 수행. Worker가 수행함 -> multi-thread로 확장 가능하도록 구성한건가?
1. update states
2. prepare inputs
3. forward pass
4. gather last-token states
5. sample

forward pass에서 kv-cache 블락들이 활용되는 다이어그램
궁금한 점은 kv-cache block과 decode 결과는 어떤 연관성이 있길래 저렇게 같은 블락 안의 빈 공간을 활용할 수 있는것인가.




















































