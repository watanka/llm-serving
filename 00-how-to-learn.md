# vLLM 학습 & 기여 로드맵

> 원칙: **전부 이해하려 하지 말고, 내 문제(Gemma4 + MTP 성능)에서 출발해 "요청 하나의 경로"를 따라간다.**
> vLLM은 수십만 줄 규모다. 지도(아키텍처) → 경로 하나(trace) → 실험(로그·디버거·테스트) → 작은 변경 순서로 간다.

## 1. 큰 코드베이스를 읽는 일반 원칙

1. **목표를 좁힌다** — "vLLM 전체"가 아니라 "MTP draft가 어떻게 검증되나" 같은 질문 하나.
2. **지도 먼저** — 공식 아키텍처 문서, 디렉터리 구조, 프로세스 구조부터 본다.
3. **요청 하나를 끝까지 따라간다** — API → EngineCore → Scheduler → Worker → Sampler → 응답.
4. **읽지 말고 실행한다** — 로그 레벨을 올리고, 브레이크포인트·print를 넣고, 테스트를 돌린다.
5. **테스트가 명세다** — `tests/v1/core/test_scheduler.py` 같은 테스트가 "의도된 동작"을 가장 짧게 보여준다.
6. **왜 이렇게 짰는지는 PR에 있다** — `git log -L` / `git blame` → PR 번호 → 리뷰 토론을 읽는다.
7. **노트에 남긴다** — 이 `notes/` 폴더에 질문 → 답 → 코드 위치 형식으로 기록한다.

## 2. vLLM 이해 순서 (추천)

| 단계 | 할 일 | 산출물 |
|---|---|---|
| 0. 큰 그림 | "Inside vLLM" 블로그 + `docs/design/arch_overview.md` | 프로세스 구조 그림 1장 |
| 1. 요청 경로 | `vllm/entrypoints/openai` → `v1/engine/async_llm.py` → `v1/engine/core.py` → `v1/core/sched/scheduler.py` → `v1/worker/gpu_model_runner.py` | 호출 흐름 노트 |
| 2. 스케줄러 | `schedule()` / `update_from_output()` + `tests/v1/core/test_scheduler.py` | ✅ 01-continuous-batching.md |
| 3. KV cache | `docs/design/paged_attention.md`, `prefix_caching.md`, `hybrid_kv_cache_manager.md` → `v1/core/kv_cache_manager.py`, `block_pool.py` | KV 노트 |
| 4. 투기 디코딩 | `docs/features/speculative_decoding/mtp.md` → `v1/spec_decode/gemma4.py`, `models/gemma4_mtp.py`, `v1/sample/rejection_sampler.py` | MTP 노트 |
| 5. 실행 최적화 | `docs/design/cuda_graphs.md`, `torch_compile.md` → `gpu_model_runner.py` | CUDA graph 노트 |
| 6. 성능 관측 | `docs/design/metrics.md`, `docs/contributing/profiling.md` | 프로파일 결과 |

### 실습 팁: 디버거로 따라가기
```bash
# 한 프로세스에서 돌게 해서 브레이크포인트가 걸리게 한다
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=DEBUG \
  .venv/bin/python my_offline_test.py   # LLM(...).generate(...) 를 부르는 작은 스크립트
```
- 작은 모델(예: Gemma4 E2B + assistant)로 `scheduler.schedule()`에 브레이크포인트를 걸고 `num_scheduled_tokens`, `scheduled_spec_decode_tokens`를 관찰한다.
- 단위 테스트는 GPU 없이도 도는 것이 많다: `pytest tests/v1/core/test_scheduler.py -k chunked`

## 3. PR까지 가는 길

### 3-1. 먼저 알아둘 vLLM 규칙
- `AGENTS.md`, `docs/contributing/README.md`, `.github/PULL_REQUEST_TEMPLATE.md`를 읽는다.
- 커밋마다 **DCO 서명 필수**: `git commit -s`
- PR 제목에 prefix를 붙인다: `[Bugfix]`, `[Perf]`, `[Core]`, `[Doc]`, `[Model]`, `[Spec Decode]` 등
- `pre-commit install` → lint, format, mypy 통과
- **AI 보조 PR 규칙 (AGENTS.md)**: 중복 PR 확인 필수, 오타 하나 같은 사소한 PR 금지, 사람이 모든 변경 줄을 이해하고 테스트를 돌려야 함, PR 본문에 AI 사용 사실·테스트 결과·(출력에 영향이 있으면) 모델 평가 결과를 명시

### 3-2. 기여 주제 찾기 — 내 업무에서 나오는 게 가장 좋다
현재 서빙(Gemma4 4bit + MTP)을 운영하면서 생기는 것들:
1. **버그 재현** — Gemma4 MTP + 양자화 조합에서 에러, 이상 동작 → 최소 재현 코드로 issue 등록 → 수정 PR
2. **성능 근거 있는 개선** — 프로파일에서 찾은 병목 + before/after 벤치마크 (vLLM 리뷰어는 수치를 요구한다)
3. **문서** — 이 노트 중 공식 문서에 없는 부분 (예: Gemma4 MTP 튜닝 가이드, `docs/features/speculative_decoding/mtp.md` 보강). 단, 의미 있는 분량으로.
4. **테스트 보강** — 내가 겪은 버그 케이스를 테스트로
5. GitHub `good first issue`, `help wanted` 라벨

### 3-3. 절차
```
1. issue 검색 + 중복 PR 확인 (gh issue/pr list)
2. issue에 댓글: 재현 결과·접근 방식 공유 → 메인테이너 반응 확인
3. fork → 브랜치 → 수정 → 관련 테스트 + pre-commit
4. 성능/출력 영향이 있으면 벤치·평가 수치 첨부
5. Draft PR → 리뷰 반영 (리뷰 왕복 여러 번이 정상)
6. vLLM Slack (#contributors 등) / discuss.vllm.ai 에서 질문
```

## 4. 링크 모음

### vLLM 전용 (최우선)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/anatomy-of-vllm) — **가장 추천.** V1 엔진, paged attention, continuous batching, prefix caching, spec decode, 분산, 벤치마크까지 한 번에 다룸
- [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — V1 설계 의도 (prefill/decode 통합 스케줄러 등)
- [공식 Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) — 프로세스 구조 (로컬: `docs/design/arch_overview.md`)
- [vLLM Contributing 가이드](https://docs.vllm.ai/en/latest/contributing/) — DCO, pre-commit, PR 제목 규칙
- [PagedAttention 논문 (SOSP 2023)](https://arxiv.org/abs/2309.06180) — KV cache 관리의 출발점
- [discuss.vllm.ai — Scheduler in vllm](https://discuss.vllm.ai/t/scheduler-in-vllm/922), [V1 Scheduler token budget 토론](https://discuss.vllm.ai/t/vllm-v1-scheduler-inconsistent-request-scheduling-under-token-budget-limit/1241) — 실제 Q&A
- [AMD ROCm 블로그: vLLM V1](https://rocm.blogs.amd.com/software-tools-optimization/vllmv1-rocm-llm/README.html) — chunked prefill 예산 배분 그림

### 한글 자료
- [위키독스 — vLLM 실습 (책)](https://wikidocs.net/book/20727) — 내부 설계, 스케줄러, 연속 배치 챕터
- [위키독스 — PagedAttention과 메모리 관리](https://wikidocs.net/337218)
- [위키독스 — Code Reading 노트](https://wikidocs.net/382458) — Spinellis 『Code Reading』 정리
- [velog — 오픈소스 기여하기 후기](https://velog.io/@coding_goat/OpenSource-오픈소스-기여-하기)
- [오픈소스 기여 가이드 (Git/GitHub)](https://braindetox.kr/posts/git_github_guide_06_opensource.html) — fork → PR 기본

### 큰 코드베이스 읽기 일반론 (영어)
- [Arpit Bhayani — Navigating Massive Codebases](https://arpitbhayani.me/videos/navigating-massive-codebases-first-principles/) — 규모 파악, 코드를 그래프로 보기
- [Holloway — How to Read Unfamiliar Code](https://holloway.com/g/junior-to-senior/sections/how-to-read-unfamiliar-code)
- [Handmade Network — How to understand a large codebase](https://git.handmade.network/how-to-understand-a-large-codebase.html)
- [Sandstorm — Understanding Unknown Source Code](https://sandstorm.de/blog/posts/understanding-unknown-source-code) — 버그·디버거 중심 접근
- [goodfirstissue.org — How to make your first open source PR](https://goodfirstissue.org/blog/posts/how-to-make-your-first-open-source-pr)
