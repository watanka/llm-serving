# vLLM 학습 & 성능 튜닝 노트

> 기준 코드: `vllm` main @ `be4ce2e505` (2026-10-14)
> 로컬에서는 vLLM 저장소 안 `notes/`로 두고 `.git/info/exclude`로 upstream과 분리해서 관리한다.

## 내 서빙 환경

| 항목 | 값 |
|---|---|
| Target 모델 | Gemma 4 (4bit 양자화) — TODO: 정확한 체크포인트명 / 양자화 방식(AWQ·GPTQ·compressed-tensors 등) |
| 투기 디코딩 | MTP, Gemma 4 assistant 모델 사용 (`"method":"mtp"`) |
| Chunked prefill | 사용 |
| KV cache / Prefix caching | 사용 |
| GPU | TODO: 종류 / 개수 / 메모리 |
| 워크로드 | TODO: 평균 입력·출력 길이, 목표 동시접속, SLA(TTFT/TPOT) |

TODO 칸을 채우면 성능 튜닝 문서의 권장값을 더 구체적으로 좁힐 수 있다.

## 목차

- [00-how-to-learn.md](00-how-to-learn.md) — 큰 코드베이스 읽는 법, vLLM 이해 순서, PR까지 가는 길, 링크 모음

### 01-understanding (구조 이해)
- [01-continuous-batching.md](01-understanding/01-continuous-batching.md) — 스케줄러가 매 step 요청을 어떻게 섞는가 (chunked prefill, MTP, preemption 포함)
- [my-note.md](01-understanding/my-note.md) — "Inside vLLM" 블로그 읽으며 정리 중인 개인 노트

### 02-performance (성능 개선)
- [01-load-test-config-tuning.md](02-performance/01-load-test-config-tuning.md) — 부하테스트로 config 찾기 (`vllm bench serve` / `vllm bench sweep`)
- [02-additional-optimizations.md](02-performance/02-additional-optimizations.md) — 그 외 적용할 만한 최적화 후보 목록
