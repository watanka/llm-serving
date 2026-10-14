# 그 외 적용할 만한 최적화 후보

부하테스트 config 튜닝(01번 문서) 외에 시도할 수 있는 것들이다. **기대효과 / 리스크 / 확인 방법**을 같이 적는다.
우선순위: ★★★ 먼저 해볼 것, ★★ 상황에 따라, ★ 실험적.

---

## A. 메모리 → 동시성 확보

### A1. KV cache FP8 양자화 ★★★
```bash
--kv-cache-dtype fp8
```
- 효과: KV 용량이 약 2배 → 같은 GPU에서 동시성↑, preemption↓, 긴 컨텍스트에 유리
- 리스크: 약간의 품질 저하 가능 → **실서비스 평가셋으로 정확도 비교 필수**
- 참고: `vllm/config/cache.py`의 `CacheDType`에 `fp8`, `fp8_per_token_head`, `int8_per_token_head`, `turboquant_*` 등 여러 옵션이 있다. GPU 아키텍처와 attention backend가 지원하는지 시작 로그에서 확인한다.
- Gemma4 assistant는 target과 KV를 공유하므로 MTP 쪽도 같이 혜택을 받는다.

### A2. `--max-model-len` 현실화 ★★★
- 실제 요청의 최대 길이 + 여유로 설정한다. 불필요하게 크면 CUDA graph, 버퍼, 최대 동시성 계산에서 손해를 본다.

### A3. `--gpu-memory-utilization` ★★
- 0.92 → 0.95 정도까지. 같은 GPU에 다른 프로세스가 없을 때만.

### A4. Hybrid KV cache manager 확인 ★★
- Gemma 계열은 sliding window + full attention이 섞인 구조다. hybrid KV cache manager가 켜져 있으면 sliding window 레이어는 window 밖 블록을 해제해 메모리를 아낀다.
- `--disable-hybrid-kv-cache-manager`를 **쓰지 않았는지** 확인한다 (기본 None = 자동).

---

## B. MTP(투기 디코딩) 최적화

### B1. 동시성별 K 튜닝 → 동적 K ★★★
```json
--speculative-config '{"method":"mtp","model":"<assistant>","num_speculative_tokens":3,
  "num_speculative_tokens_per_batch_size":[[1,8,3],[9,32,2],[33,256,1]]}'
```
- 배치 크기 구간별로 K를 다르게 쓴다 (`vllm/config/speculative.py`의 `num_speculative_tokens_per_batch_size`, 형식 `(start, end, K)`).
- 구간 경계는 01번 문서 Step 6 측정 결과로 정한다.
- 리스크: 비교적 새 기능이다. 버전별 지원 여부와 Gemma4 MTP 조합 동작을 먼저 확인한다.

### B2. acceptance rate 모니터링 ★★★
- `Mean acceptance length`가 1 + K에 가까울수록 좋다. 위치별 acceptance(`..._per_pos`)에서 뒤쪽이 0에 가깝다면 K를 줄인다.
- **temperature가 높으면 acceptance가 떨어진다.** 서비스 sampling 설정으로 측정한다.

### B3. draft 샘플링 방식 ★
- `draft_sample_method`: `greedy`(기본) vs `probabilistic`. 고온 샘플링 서비스라면 probabilistic이 acceptance를 높일 수 있지만 메모리를 더 쓴다.
- `rejection_sample_method`: `standard` / `block` — `block`은 draft 묶음을 한 번에 검증하는 방식. 실험해볼 가치가 있다.

### B4. assistant 모델 양자화 일관성 ★★
- target이 4bit일 때 assistant가 bf16이면 draft 비용이 상대적으로 커질 수 있다. 다만 assistant는 작아서 영향이 제한적이다. 프로파일로 draft 시간 비중을 확인한 뒤 판단한다.

---

## C. 스케줄링 / CPU 오버헤드

### C1. Async scheduling 켜졌는지 확인 ★★★
- `--async-scheduling` (기본 자동). GPU가 step N을 도는 동안 CPU가 step N+1을 스케줄해서 step 사이 공백을 없앤다.
- 시작 로그와 config 출력에서 `async_scheduling=True`인지 본다. 투기 디코딩과의 호환 여부는 버전마다 다르므로 꺼져 있다면 이유를 로그에서 확인한다.

### C2. `--stream-interval` ★★
- 기본 1(토큰마다 전송). 2~5로 올리면 detokenize/전송 오버헤드가 줄어 고동시성 처리량↑. 대신 스트리밍이 덜 부드러워진다.

### C3. API 서버 프로세스 분리/증설 ★★
- 고동시성에서는 HTTP·토크나이즈·detokenize가 병목이 될 수 있다. `--api-server-count`로 프론트엔드 프로세스를 늘리는 것을 검토한다.

### C4. 입장 제어로 tail latency 보호 ★★
- `--max-num-queued-reqs`, `--max-num-queued-tokens`: 한계 넘으면 503으로 거절 → 로드밸런서가 다른 인스턴스로 보냄
- `--watermark 0.01~0.05`: 여유 KV를 남겨 preemption 폭주 방지

---

## D. 커널 / 그래프

### D1. CUDA graph 캡처 범위 ★★
- `--compilation-config '{"cudagraph_capture_sizes":[...]}'` 또는 `max_cudagraph_capture_size`
- 실제 동시성 × (1+K) 구간이 캡처 범위 안에 들어가야 decode step이 graph로 실행된다. 범위를 벗어나면 eager 실행으로 느려진다.
- `--enforce-eager`를 쓰고 있지 않은지 확인한다 (디버깅용).

### D2. 4bit 양자화 커널 확인 ★★★
- 같은 4bit라도 커널에 따라 속도 차이가 크다 (예: AWQ/GPTQ → Marlin 계열, compressed-tensors W4A16 등).
- 시작 로그에서 실제 선택된 커널/quant method를 확인한다. 더 빠른 커널로 자동 변환되지 않았다면 원인을 찾는다.
- GPU가 FP8/FP4를 지원한다면(Hopper/Blackwell) **W8A8-FP8, NVFP4 체크포인트**가 4bit weight-only보다 고동시성에서 빠를 수 있다. 품질과 속도를 같이 비교한다.

### D3. Attention backend ★
- `--attention-backend` 인자(FLASH_ATTN / FLASHINFER 등)로 비교한다 (현재 버전은 환경변수가 아니라 CLI 인자). Gemma의 sliding window, softcap 지원 여부에 따라 선택지가 제한될 수 있다.

---

## E. 워크로드 레벨

### E1. Prefix caching 극대화 ★★★
- 기본 켜짐. **시스템 프롬프트·few-shot을 요청 맨 앞에 고정**하면 hit이 크게 오른다 (블록 단위 해시라서 앞부분이 1토큰만 달라도 그 뒤는 모두 miss).
- 날짜·사용자명 같은 가변 값을 프롬프트 **앞쪽에 두지 않는다.**
- `vllm:prefix_cache_hits` / `queries`로 hit rate를 확인한다.

### E2. 멀티 인스턴스 / 병렬화 ★★
- 4bit 모델이 GPU 1장에 들어가면 TP를 늘리기보다 **인스턴스 복제(DP 또는 별도 서버 + LB)** 가 처리량 면에서 대체로 유리하다.
- 로드밸런서는 prefix-aware 라우팅을 쓰면 캐시 hit이 유지된다.

### E3. 우선순위 스케줄링 ★
- `--scheduling-policy priority`: 인터랙티브와 배치 요청이 섞여 있다면 요청에 priority를 붙여 분리한다.

---

## F. 프로파일링 (병목 확인)

- `vllm bench latency` — 단일 배치 지연시간
- PyTorch profiler: `--profiler-config` 설정 후 `/start_profile`, `/stop_profile` → step별 GPU 시간 분해 (target forward / draft / sampling / CPU gap)
- Nsight Systems로 step 사이 CPU 공백 확인 → async scheduling 효과 검증

---

## 체크리스트

- [ ] 현재 운영 설정·버전·시작 로그 저장 (non-default args, KV cache size, quant kernel)
- [ ] 실서비스 유사 데이터셋 준비
- [ ] 베이스라인 곡선 (동시성 스윕)
- [ ] A1 KV fp8 품질/성능 비교
- [ ] 01번 Step 3~6 스윕
- [ ] B1 동적 K
- [ ] C1 async scheduling 확인
- [ ] D2 양자화 커널 확인
- [ ] E1 프롬프트 구조 점검
