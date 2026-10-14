# 부하테스트로 Config 찾기

## 0. 목표부터 정한다

"성능 최대치"는 하나가 아니다. 먼저 무엇을 최적화할지 고른다.

| 목표 | 핵심 지표 | 주로 만지는 손잡이 |
|---|---|---|
| 체감 속도 (채팅) | TTFT p99, TPOT/ITL p99 | `max_num_batched_tokens` ↓, `num_speculative_tokens` |
| 처리량 (배치·대량 처리) | output tok/s, req/s | `max_num_seqs` ↑, `max_num_batched_tokens` ↑ |
| **SLA 지키면서 최대 처리량** (권장) | goodput | 위 전부 + 동시성 스윕 |

> 권장 방식: **SLA(예: TTFT p99 < 1s, TPOT p99 < 50ms)를 먼저 정하고, 그 SLA를 만족하는 최대 동시성/처리량(goodput)을 찾는다.**

## 1. 측정 도구

### 1-1. 단일 측정: `vllm bench serve`
서버를 띄운 상태에서 클라이언트로 부하를 준다 (`vllm/benchmarks/serve.py`).

```bash
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model <서빙중인-모델명> \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 500 \
  --max-concurrency 32 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --goodput ttft:1000 tpot:50 \
  --save-result --result-dir ./bench-results
```

- `--max-concurrency`: 동시에 떠 있는 요청 수 (실서비스 동시접속을 흉내)
- `--request-rate`(+ `--burstiness`): 초당 요청 도착률 (포아송 도착)
- `--goodput`: SLA를 만족한 요청만 센 처리량
- `--ramp-up-strategy`: 부하를 점점 올려가며 한계점 탐색

### 1-2. 자동 스윕: `vllm bench sweep`
`docs/benchmarking/sweeps.md` 참고. 서버 설정 × 벤치 설정을 데카르트 곱으로 돌려준다.

- `vllm bench sweep serve` — 서버 파라미터 조합마다 서버를 다시 띄우고 측정
- `vllm bench sweep serve_workload` — 동시성/요청률을 올려가며 latency-throughput 곡선 탐색
- `vllm bench sweep plot` / `plot_pareto` — 결과 시각화, Pareto front

## 2. 데이터셋을 실제처럼 맞춘다 (가장 중요)

- MTP는 **텍스트 성격에 따라 acceptance rate가 크게 달라진다.** `random` 데이터셋(무작위 토큰)은 draft가 거의 맞지 않아 **MTP 효과를 과소평가**한다.
- 반드시 **실제 서비스 프롬프트 샘플**(개인정보 제거)이나 ShareGPT 같은 실제 대화 데이터로도 측정한다.
  - `--dataset-name custom --dataset-path my_prompts.jsonl` 또는 `sharegpt`
- 입력/출력 길이 분포(평균·p90)를 실제 로그에서 뽑아서 `random` 시나리오도 함께 만든다.

## 3. 튜닝 순서 (한 번에 하나씩)

### Step 1. 베이스라인
현재 운영 설정 그대로 동시성 `1, 4, 8, 16, 32, 64, 128`을 스윕해 latency-throughput 곡선을 만든다.
→ 동시성 1의 TPOT = 순수 단일 요청 속도. MTP 효과는 여기서 가장 크게 보인다.

### Step 2. KV cache 용량 확보
서버 시작 로그에서 확인:
- `GPU KV cache size: N tokens`
- `Maximum concurrency for X tokens per request: Y`

조정:
- `--gpu-memory-utilization` (기본 0.92) → OOM 안 나는 선에서 올림 (0.92~0.95)
- `--max-model-len` → 실제 필요한 길이로 **줄이면** 남는 메모리가 KV로 간다
- (선택) `--kv-cache-dtype fp8` → KV 용량 거의 2배. 품질 검증 필요 (02-additional-optimizations.md)

**판정 기준**: 목표 동시성에서 `vllm:num_preemptions`가 0에 가까워야 한다. preemption이 계속 나면 KV 용량이 모자란 것이다.

### Step 3. `max_num_seqs` (동시 처리 요청 수)
- KV cache가 감당 가능한 최대 동시성 근처로 잡는다 (Step 2 로그의 `Maximum concurrency`).
- 너무 크면 preemption과 decode step 지연(TPOT↑)이 생기고, 너무 작으면 waiting 큐가 쌓여 TTFT↑, 처리량↓.
- 후보: `32, 64, 128, 256`
- CUDA graph 크기는 유지한 채 running 수만 줄이고 싶으면 `--max-num-active-seqs`.

### Step 4. `max_num_batched_tokens` (step당 토큰 예산)
continuous batching 문서 5절의 핵심 트레이드오프.

| 값 | 효과 |
|---|---|
| 작게 (예: 2048) | prefill chunk가 작아져 decode가 덜 끊김 → **TPOT/ITL 안정**, 대신 긴 프롬프트의 TTFT↑ |
| 크게 (예: 8192~16384) | prefill을 빨리 끝냄 → **TTFT↓, 처리량↑**, 대신 prefill 섞인 step에서 ITL 튐 |

- 후보: `2048, 4096, 8192, 16384`
- `max_num_seqs × (1 + num_speculative_tokens)`보다는 충분히 커야 decode만으로 예산이 꽉 차지 않는다.

### Step 5. 긴 프롬프트가 섞이는 경우
- `--long-prefill-token-threshold` (예: 예산의 1/4~1/2): 긴 prefill 하나가 step을 독점하지 못하게 한다.
- `--long-prefill-token-threshold-adaptive`: 요청이 적을 때는 공정 몫까지 허용한다.

### Step 6. MTP `num_speculative_tokens` (K)
**동시성 구간별로 최적 K가 다르다.**
- 낮은 동시성: GPU가 놀고 있으므로 K를 키울수록(2~4) 이득
- 높은 동시성: 이미 compute bound. draft 검증 비용이 이득을 잡아먹어 K=1 또는 0(MTP 끔)이 나을 수 있음
- 측정 후보: `K = 1, 2, 3, 4` × 동시성 `1, 8, 32, 64, 128`
- 동시성별로 K가 갈린다면 → `num_speculative_tokens_per_batch_size`(동적 K, 02-additional 문서 참고)

## 4. 스윕 예시

`notes/02-performance/sweep/serve_params.json`
```json
[
  {"_benchmark_name": "s64_b4k_k1",  "max_num_seqs": 64,  "max_num_batched_tokens": 4096,
   "speculative_config": "{\"method\":\"mtp\",\"model\":\"<assistant>\",\"num_speculative_tokens\":1}"},
  {"_benchmark_name": "s64_b4k_k2",  "max_num_seqs": 64,  "max_num_batched_tokens": 4096,
   "speculative_config": "{\"method\":\"mtp\",\"model\":\"<assistant>\",\"num_speculative_tokens\":2}"},
  {"_benchmark_name": "s128_b8k_k1", "max_num_seqs": 128, "max_num_batched_tokens": 8192,
   "speculative_config": "{\"method\":\"mtp\",\"model\":\"<assistant>\",\"num_speculative_tokens\":1}"},
  {"_benchmark_name": "s128_b8k_k2", "max_num_seqs": 128, "max_num_batched_tokens": 8192,
   "speculative_config": "{\"method\":\"mtp\",\"model\":\"<assistant>\",\"num_speculative_tokens\":2}"}
]
```
> JSON 값을 CLI 인자로 바꾸는 방식은 `vllm bench sweep serve --dry-run`으로 먼저 확인한다.

```bash
vllm bench sweep serve_workload \
  --serve-cmd 'vllm serve <gemma4-4bit-model> --gpu-memory-utilization 0.92 --max-model-len <N>' \
  --bench-cmd 'vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model <gemma4-4bit-model> --dataset-name sharegpt --dataset-path <path> --num-prompts 500' \
  --serve-params notes/02-performance/sweep/serve_params.json \
  --workload-var max_concurrency \
  --output-dir notes/02-performance/results \
  --experiment-name gemma4-mtp-v1 \
  --dry-run
```
`--dry-run`으로 명령을 확인한 뒤 떼고 실행한다.

## 5. 측정 중 같이 봐야 할 서버 지표 (`/metrics`)

| 메트릭 | 의미 | 이상 신호 |
|---|---|---|
| `vllm:kv_cache_usage_perc` | KV 사용률 | 계속 ~100% → KV 부족 |
| `vllm:num_preemptions` | preemption 누적 | 증가 → `max_num_seqs`↓ 또는 KV↑ |
| `vllm:num_requests_waiting` | 대기 큐 | 계속 증가 → 용량 초과 |
| `vllm:spec_decode_num_accepted_tokens` / `num_draft_tokens` | MTP acceptance | 낮으면 K↓ |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | 위치별 acceptance | 뒤쪽 위치가 거의 0이면 K가 과함 |
| `vllm:prefix_cache_hits` / `queries` | prefix cache hit | 시스템 프롬프트가 공통인데 낮으면 점검 |

서버 로그에도 주기적으로 `SpecDecoding metrics: Mean acceptance length ...`가 찍힌다.

## 6. 결과 기록 템플릿

| 실험명 | max_num_seqs | batched_tokens | K | 동시성 | TTFT p99 | TPOT p99 | out tok/s | goodput | preempt | accept len | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | | | | | | |
