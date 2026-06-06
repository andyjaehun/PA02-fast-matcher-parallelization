# PA02 — Fast Correlative Scan Matcher 병목 분석 및 고속화

FastMatcher 함수별 병목 프로파일링 및 CPU/GPU 선택 최적화

---

## 실험 환경

- **장비**: NVIDIA Jetson Nano Developer Kit
- **CPU**: ARM Cortex-A57, 4코어
- **GPU**: Maxwell SM 5.3, 128 CUDA cores
- **OS**: Ubuntu 18.04 (Docker: dustynv/ros:melodic-ros-base-l4t-r32.7.1)
- **CUDA**: 10.2 (compute capability 5.3)

---

## 파일 구조

```
src/
├── score_all_pa02_final.cpp       # PA01 최종 GPU 커널 (grid 상주 + warp shuffle + block128)
├── score_all_pa02_batch.cpp       # PA02 신규 GPU 배치 커널 (scan 전체 한 번에 처리)
│
├── fast_matcher_profiled.cpp      # Step1: 함수별 chrono 타이머 (병목 측정)
├── fast_matcher_batch_score.cpp   # 최적화1: Score GPU 배치화 (15→1회 GPU 호출)
├── fast_matcher_cpu_fallback.cpp  # 최적화2: Branch CPU Adaptive 전환
├── fast_matcher_opt_scans.cpp     # 최적화3: MakeScans 연산 최적화
├── fast_matcher_opt_lowcands.cpp  # 최적화4: MakeLowCands 알고리즘 개선
└── fast_matcher_combined.cpp      # 최종 통합: 모든 최적화 적용

cmake/
├── CMakeLists_profiling.txt       # Step1 빌드
├── CMakeLists_batch_score.txt     # 최적화1 단독 빌드
├── CMakeLists_cpu_fallback.txt    # 최적화2 단독 빌드
├── CMakeLists_opt_scans.txt       # 최적화3 단독 빌드
├── CMakeLists_opt_lowcands.txt    # 최적화4 단독 빌드
└── CMakeLists_combined.txt        # 최종 통합 빌드
```

---

## Step 1 — Baseline Profiling

`fast_matcher_profiled.cpp` (chrono 타이머) + `score_all_pa02_final.cpp` (PA01 GPU 최종)

| 함수 | 시간 | 비율 |
|------|------|------|
| Score(coarse) | 71.04 ms | 97.6% |
| MakeScans | 0.743 ms | 1.0% |
| MakeLowCands | 0.656 ms | 0.9% |
| Branch | 0.379 ms | 0.5% |
| MakeBounds | 0.001 ms | ~0% |
| **Total** | **72.82 ms** | **100%** |

---

## Step 2/3 — 고속화 대상 및 CPU/GPU 선택 근거

### Score(coarse) + Branch → GPU 배치화 + CPU Adaptive

> **전제**: PA01에서 `score_all()`은 이미 GPU로 가속화된 상태 (`score_all_pa02_final.cpp` 링크)  
> PA02 baseline = 원본 fast_matcher 구조 + PA01 GPU 최종 score_all

**Score(coarse) 구조 문제:**

```cpp
for s in 0..14:                        ← 15개 scan angle 순차
    score_all(px_s, py_s, cx_s, cy_s) ← GPU 호출  (PA01 최적화 커널)
    → h2d(px/py/cx/cy) + kernel + d2h ← 매번 반복

GPU 호출 1회 비용 분석:
  h2d (데이터 전송):    ~0.30 ms
  kernel (실제 연산):   ~0.87 ms
  d2h (결과 수신):      ~0.11 ms
  합계:                 ~1.28 ms × 15회 = 19.2 ms
```

⭐ 15회의 반복적인 GPU launch 가 핵심 비용

**Branch 내부 Score 구조 문제:**

```
Branch 재귀 → Score(cands=4) 반복 호출
→ GPU 호출 시: h2d+d2h overhead = 0.47 ms
→ 실제 kernel 연산:             = 0.01 ms (4개 × 1081 pts)
→ overhead가 실제 연산의 47배
```

⭐ GPU를 쓸수록 오히려 느려지는 구조

### MakeScans → CPU 연산 최적화

| 방법 | 결과 | 이유 |
|------|------|------|
| OpenMP outer loop | 2.12 ms ↑ | 스레드 생성 비용 > 15회 병렬화 이득 |
| GPU 커널 | 1.63 ms ↑ | CUDA launch overhead > 실제 연산 |
| **CPU 연산 최적화** | **0.127 ms ↓** | 알고리즘 개선 |

### MakeLowCands → CPU 알고리즘 개선

- `make_cand` 15번 중복 호출 → local search에서 모든 scan의 bounds 동일
- `reserve()` 없음 → 15,360번 push_back 중 reallocation 반복

---

## Step 4 — 구현 및 결과

### 최적화1: Score GPU 배치화 (`fast_matcher_batch_score.cpp`)

15개 scan의 px/py/cx/cy를 하나의 배열로 합산 → `score_all_batch()` 1회 호출

| | Baseline | Score GPU 배치 |
|--|---------|--------------|
| Score(coarse) | 71.04 ms | **14.35 ms** |
| GPU 호출 횟수 | 15회 | **1회** |
| 개선 | | **4.95x** |

### 최적화2: Branch CPU Adaptive 전환 (`fast_matcher_cpu_fallback.cpp`)

```cpp
if (cand->size() <= 32) {
    // CPU 직접 계산 — GPU overhead 완전 제거
} else {
    // GPU 배치 (Score coarse용)
    score_all_batch(...)
}
```

| | Score GPU 배치 | +Branch CPU 전환 |
|--|--------------|----------------|
| Branch 시간 | 1.77 ms | **0.34 ms** |
| 개선 | | **5.2x** |

### 최적화3: MakeScans 연산 최적화 (`fast_matcher_opt_scans.cpp`)

- `(init.x - ox_) / res_` → 루프 밖으로 호이스팅
- double → float 전환 (xs/ys가 float이므로)
- 나눗셈 32,430번 → `inv_res` 곱셈으로 대체
- `#pragma GCC optimize("O3", "unroll-loops", "unsafe-math-optimizations")`

| | Baseline | MakeScans 최적화 |
|--|---------|----------------|
| MakeScans | 0.743 ms | **0.127 ms** |
| 개선 | | **5.8x** |

### 최적화4: MakeLowCands 알고리즘 개선 (`fast_matcher_opt_lowcands.cpp`)

- `make_cand` 1번만 호출 후 15개 scan에 재사용
- `out.reserve(15 × base_cx.size())` 로 reallocation 제거

| | Baseline | MakeLowCands 최적화 |
|--|---------|-------------------|
| MakeLowCands | 0.656 ms | **0.153 ms** |
| 개선 | | **4.3x** |

---

## Step 5 — 최종 비교 분석 (10회 평균)

`fast_matcher_combined.cpp` + `score_all_pa02_batch.cpp`

| 함수 | CPU Baseline | 최종 통합 | 개선 |
|------|-------------|----------|------|
| Score(coarse) | 71.04 ms | 14.94 ms | 4.75x |
| Branch | 0.379 ms | 0.322 ms | 1.2x |
| MakeScans | 0.743 ms | 0.286 ms | 2.6x |
| MakeLowCands | 0.656 ms | 0.361 ms | 1.8x |
| **Total** | **72.82 ms** | **15.91 ms** | **4.58x** |

---

## 빌드 및 실행 방법

### Jetson에서 (기본 측정 — 개별 최적화 검증)

```bash
# 1. 파일 수정 (필요시)
nano ~/catkin_ws/src/cartographer_parallel/src/score_all_pa02_batch.cpp

# 2. CMakeLists 교체 후 빌드
cp cmake/CMakeLists_combined.txt CMakeLists.txt
cd ~/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release && source devel/setup.bash

# 3. 측정 파일 초기화 후 rosbag 실행
rm -f /tmp/pa02_combined_runtime.csv
roslaunch cartographer_parallel cartographer_parallel_with_bag.launch \
  ns:="student_06" \
  linear_search_window:=3.1 \
  branch_and_bound_depth:=3 \
  2>&1 | grep "pa02_combined"

# 4. CSV에서 1회 평균 계산
awk -F',' 'NR>1 {sc+=$2; lc+=$3; s+=$4; b+=$5; t+=$6; n++} END \
  {print "scans=" sc/n "ms  lowcands=" lc/n "ms  score=" s/n "ms  branch=" b/n "ms  total=" t/n "ms"}' \
  /tmp/pa02_combined_runtime.csv
```

> 개별 최적화 검증 시에는 `CMakeLists_combined.txt` 대신 해당 최적화의 CMakeLists로 교체  
> (예: `cmake/CMakeLists_batch_score.txt`, `cmake/CMakeLists_opt_scans.txt` 등)

### 최종 비교 — 10회 반복 측정 (위 기본 측정을 10번 자동 반복)

```bash
rm -f /tmp/all_runs.txt

for i in $(seq 1 10); do
  echo "=== Run $i/10 ==="
  rm -f /tmp/pa02_combined_runtime.csv

  timeout 110 roslaunch cartographer_parallel \
    cartographer_parallel_with_bag.launch \
    ns:="student_06" \
    linear_search_window:=3.1 \
    branch_and_bound_depth:=3 \
    2>/dev/null || true

  awk -F',' 'NR>1 {sc+=$2; lc+=$3; s+=$4; b+=$5; t+=$6; n++} END \
    {print sc/n, lc/n, s/n, b/n, t/n}' \
    /tmp/pa02_combined_runtime.csv >> /tmp/all_runs.txt

  echo "Run $i 완료"
done

echo "=== 10회 평균 ==="
awk '{sc+=$1; lc+=$2; s+=$3; b+=$4; t+=$5; n++} END \
  {printf "scans=%.3fms  lowcands=%.3fms  score=%.3fms  branch=%.3fms  total=%.3fms\n", \
  sc/n, lc/n, s/n, b/n, t/n}' /tmp/all_runs.txt
```

> bag 1회 재생 약 96초 → 10회 약 16분 소요. `timeout 110`으로 자동 종료 처리됨

---

## 결론

**CPU 기본 대비 4.58배 속도 향상 달성** (72.82 ms → 15.91 ms, 10회 평균)

| 최적화 | 절감 시간 | 기여 비율 |
|--------|----------|----------|
| Score GPU 배치화 | 56.1 ms | 98.6% |
| MakeScans 최적화 | 0.457 ms | 0.8% |
| MakeLowCands 개선 | 0.295 ms | 0.5% |
| Branch CPU 전환 | 0.057 ms | 0.1% |
| **총 절감** | **56.9 ms** | |

> **핵심 교훈:** "병렬화는 workload가 충분히 클 때만 효과적이다"  
> GPU든 OpenMP든, overhead < 연산 시간 조건이 충족될 때만 의미 있다.  
> 병목은 함수 자체가 아닌 **호출 구조**에 있을 수 있다 — Score GPU 배치화(15→1회)는 커널을 바꾼 것이 아니라 호출 횟수를 줄인 구조적 개선이었다.
