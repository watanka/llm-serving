# Continuous Batching 동작 방식

## 1. 한 줄 요약

**"요청 단위가 아니라 step(iteration) 단위로 배치를 다시 짠다."**
매 forward pass 직전에 스케줄러가 "이번 step에 어떤 요청의 토큰 몇 개를 GPU에 넣을지"를 새로 정한다.
끝난 요청은 즉시 빠지고, 대기 중인 요청은 빈자리가 생기는 즉시 들어온다.

```
Static batching:   [A A A A A . . .]   ← 가장 긴 요청이 끝날 때까지 배치 고정, GPU 놀음
                   [B B B . . . . .]
                   [C C C C C C C C]

Continuous:  step1 [A B C]
             step2 [A B C]
             step3 [A C D]      ← B 종료 → 바로 D 투입
             step4 [A C D E]    ← 여유 생기면 E 투입
```

## 2. 전체 흐름 (코드 위치)

```
API 서버 (vllm/entrypoints/openai/...)
   │  요청 → EngineCoreRequest
   ▼
EngineCore.run_busy_loop()          vllm/v1/engine/core.py:1547
   └ step()                         vllm/v1/engine/core.py:657
       1. scheduler.schedule()      → SchedulerOutput (요청별 토큰 수, 새 KV 블록)
       2. model_executor.execute_model()   → GPU forward (+ MTP draft)
       3. sample_tokens()           → 샘플링 / rejection sampling
       4. scheduler.update_from_output()   → 토큰 반영, 종료 판정, 블록 해제
```

- 스케줄러 본체: `vllm/v1/core/sched/scheduler.py` (`Scheduler.schedule()` 591행)
- async scheduling 사용 시: `vllm/v1/core/sched/async_scheduler.py` — step N의 GPU 연산 중에 step N+1을 미리 스케줄해서 CPU 공백을 줄인다.
- pipeline parallel 등에서는 `step_with_batch_queue()`(core.py:698)로 여러 배치를 겹쳐 돌린다.

## 3. 핵심 아이디어: "prefill/decode 단계 구분이 없다"

`schedule()` 맨 위 주석(scheduler.py:593)이 설계 철학을 그대로 말해준다.

- 각 요청은 두 숫자만 가진다.
  - `num_computed_tokens`: KV cache에 이미 계산되어 있는 토큰 수
  - `num_tokens_with_spec` = 프롬프트 + 지금까지 생성한 출력 + **draft(spec) 토큰**
- 스케줄러는 매 step마다 `num_computed_tokens`가 `num_tokens_with_spec`을 따라잡도록 토큰을 배정할 뿐이다.

그래서 아래가 전부 **같은 식 하나**로 표현된다.

| 상황 | 이번 step에 배정되는 토큰 |
|---|---|
| 새 요청 prefill | 프롬프트 전체 (또는 예산만큼 잘라서 = **chunked prefill**) |
| prefix cache hit | 캐시된 부분은 `num_computed_tokens`에 이미 포함 → 나머지만 |
| 일반 decode | 1개 |
| MTP 투기 디코딩 | 1 + K개 (K = `num_speculative_tokens`, 검증할 draft 토큰) |
| preemption 후 재개 | 다시 계산해야 하는 토큰 전부 |

## 4. 한 step의 스케줄링 알고리즘

### 4-1. 예산(budget)

```python
token_budget = max_num_scheduled_tokens          # 기본 = max_num_batched_tokens
input_budget = max_num_batched_tokens
draft_slots  = spec.max_num_new_slots_for_drafting   # 요청당 drafting 추가 슬롯
```
(scheduler.py:611-614)

- **`max_num_batched_tokens`**: 한 step에 GPU로 들어가는 토큰 총량 상한 → step 지연시간과 직결
- **`max_num_seqs`**: 한 step에 동시에 들어갈 수 있는 요청 수 상한
- 투기 디코딩 방식에 따라 요청당 `draft_slots`만큼 input budget을 추가로 쓴다.
  Gemma4 MTP처럼 draft를 순차 생성하는 방식은 대체로 0이지만, parallel drafting 계열(DFlash 등)은 K 또는 K-1만큼 소모한다 (speculative.py:1888 표 참고).

### 4-2. 1단계 — RUNNING 요청 먼저 (scheduler.py:657~)

`self.running` 리스트를 앞에서부터 돌면서:

1. `num_new_tokens = num_tokens_with_spec + placeholders - num_computed_tokens`
2. `long_prefill_token_threshold`가 설정돼 있으면 그 값으로 자름 (긴 prefill 하나가 예산을 독식하지 못하게)
3. 남은 `token_budget`, `max_model_len`으로 다시 자름 → **이게 chunked prefill**
4. `kv_cache_manager.allocate_slots()`로 필요한 KV 블록 확보
   - **실패하면 preemption**: 우선순위가 가장 낮은 running 요청(FCFS면 `running[-1]`, 가장 늦게 들어온 것)을 쫓아내고 다시 시도
   - 쫓겨난 요청은 KV 블록을 반납하고 waiting 큐 맨 앞으로 간다 (`_preempt_request`, 1571행). 나중에 **처음부터 다시 계산(recompute)** 한다. 단, prefix caching이 켜져 있으면 해제된 블록이 아직 남아 있는 한 캐시 hit으로 상당 부분을 되살린다.
5. 성공하면 예산에서 차감하고, draft 토큰이 있으면 `scheduled_spec_decode_tokens`에 기록

> decode 중인 요청을 먼저 챙기는 이유: 이미 KV를 들고 있는 요청이 끝나야 메모리가 돌고, 사용자 입장에서 스트리밍이 끊기지 않기 때문이다 (TPOT 보호).

### 4-3. 2단계 — WAITING 요청 투입 (scheduler.py:899~)

**이번 step에 preemption이 한 번도 없었을 때만** 실행된다 (`if not preempted_reqs`). 메모리가 부족한 상황에서 새 요청을 더 받으면 thrashing이 나기 때문이다.

`token_budget > 0`이고 running 수가 `max_num_seqs`(또는 `max_num_active_seqs`) 미만인 동안:

1. 큐 맨 앞 요청을 peek
2. **Prefix cache 조회** (`_get_local_prefix_cache_hit`) → 이미 있는 블록은 재사용하고 그만큼 `num_computed_tokens`를 올림
3. `num_new_tokens = num_tokens - num_computed_tokens`
4. `long_prefill_token_threshold`와 남은 예산으로 잘라서 첫 chunk 결정
   - chunked prefill이 꺼져 있고 프롬프트가 예산보다 크면 → 여기서 중단
5. KV 블록 할당. `scheduler_reserve_full_isl=True`(기본)면 **첫 chunk가 아니라 프롬프트 전체가 들어갈 공간이 있는지** 확인한 뒤 받는다. chunked prefill 때문에 과하게 받아놓고 나중에 줄줄이 preempt되는 걸 막기 위함이다.
6. 성공 → `running`에 추가

#### MTP 관련 디테일: decode 패딩
새로 들어온 요청이 1토큰짜리 decode 상태이고 이번 step에 prefill이 없으면, `1 + num_spec_tokens`로 **패딩**해서 모든 요청의 query 길이를 맞춘다 (1131~1153행). 배치가 "uniform decode" 모양이 되어야 **full CUDA graph**를 그대로 재사용할 수 있기 때문이다.

### 4-4. step 이후 — `update_from_output()` (2029행)

- 샘플된 토큰을 요청에 붙인다. MTP라면 rejection sampler가 **accept한 draft 수 + 1(bonus)** 만큼 한 번에 붙는다.
- reject된 draft 위치의 KV는 다음 step에 덮어쓴다 (`num_computed_tokens`를 되돌림).
- stop 조건(EOS, max_tokens, stop string)을 만족하면 종료 → KV 블록 반납 → **다음 step에서 waiting 요청이 바로 그 자리를 쓴다.** 이것이 continuous batching의 핵심 순환 고리다.

## 5. 내 설정(Gemma4 + MTP + chunked prefill + KV cache)에서 한 step의 모습

예: `max_num_batched_tokens=4096`, `num_speculative_tokens=2`, decode 중인 요청 30개, 새 요청 1개(프롬프트 6000토큰)

```
running 30개 × (1 + 2 draft) = 90 토큰
남은 예산 4096 - 90 = 4006
새 요청: prefix cache hit 1024 → 남은 4976 중 4006만 이번 step (chunk 1)
                                   다음 step에 나머지 970 (chunk 2)
--------------------------------------------------------------------
이번 step 총 4096 토큰 = decode 90 + prefill chunk 4006  (한 forward에 섞임)
```

- decode만 있으면 메모리 대역폭 bound, prefill은 compute bound → **섞어야 GPU를 고르게 쓴다.** chunked prefill의 존재 이유다.
- 그런데 이 step은 prefill chunk 때문에 무거워져서 decode 30개의 이번 토큰(TPOT)이 늦어진다. → `max_num_batched_tokens`가 **TTFT ↔ TPOT 트레이드오프의 핵심 손잡이**인 이유다 (성능 문서 참고).
- Gemma4 assistant(MTP)는 **target 모델과 KV cache를 공유**한다(`vllm/v1/spec_decode/gemma4.py` 상단 docstring). 그래서 draft 모델이 KV 메모리를 따로 크게 잡아먹지 않는다.

## 6. 관련 설정 빠른 참조

| 설정 (CLI) | 기본값 | 역할 | 정의 위치 |
|---|---|---|---|
| `--max-num-batched-tokens` | 하드웨어/사용 맥락별 자동 | step당 토큰 상한 | `vllm/config/scheduler.py:48` |
| `--max-num-seqs` | 자동 | step당 요청 수 상한 | `scheduler.py:62` |
| `--max-num-active-seqs` | None | running 입장 수만 따로 제한 (CUDA graph 크기는 유지) | `scheduler.py:68` |
| `--enable-chunked-prefill` | True | prefill 분할 허용 | `scheduler.py:134` |
| `--long-prefill-token-threshold` | 0 (끔) | 긴 prefill 한 번에 먹는 양 제한 | `scheduler.py:79` |
| `--scheduler-reserve-full-isl` | True | 입장 시 프롬프트 전체 KV 공간 확인 | `scheduler.py:190` |
| `--watermark` | 0.0 | 입장 시 남겨둘 여유 KV 블록 비율 | `scheduler.py:196` |
| `--async-scheduling` | 자동 | CPU 스케줄링과 GPU 실행 겹치기 | `scheduler.py:208` |
| `--scheduling-policy` | fcfs | fcfs / priority | `scheduler.py:159` |
| `--enable-prefix-caching` | True | 블록 단위 KV 재사용 | `vllm/config/cache.py:141` |

> 실제 적용된 기본값은 서버 시작 로그의 `non-default args`와 config 출력에서 확인한다.

## 7. 더 읽어볼 코드 (다음 이해 노트 후보)

- `vllm/v1/core/kv_cache_manager.py` — `allocate_slots`, 블록 단위 PagedAttention 메모리 관리
- `vllm/v1/core/block_pool.py` — prefix caching 해시/LRU 축출
- `vllm/v1/spec_decode/gemma4.py`, `vllm/model_executor/models/gemma4_mtp.py` — MTP draft 생성
- `vllm/v1/sample/rejection_sampler.py` — draft 검증
- `vllm/v1/worker/gpu_model_runner.py` — 입력 텐서 준비, CUDA graph 디스패치
