// PA02: Score() GPU 배치화용 score_all
// 기존: score_all() 15번 개별 호출
// 변경: score_all_batch() 1번 호출로 15개 scan 전체 처리
//
// 새 GPU 커널: 각 candidate가 자신의 scan_id로 올바른 px/py 구간 찾아서 계산
// grid 상주 + Warp Shuffle Reduction + Block 128 유지

#include "cartographer_parallel/assignment.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace cartographer_parallel {
namespace {

const char* kLogPath = "/tmp/pa02_batch_score_all_runtime.csv";

void WriteLog(int call, int n_total, int n_scans, int p,
              double total_ms, float h2d_ms, float kernel_ms, float d2h_ms) {
  static bool header = false;
  std::ofstream f(kLogPath, std::ios::app);
  if (!f) return;
  if (!header) {
    f << "call,total_ms,h2d_ms,kernel_ms,d2h_ms,n_total,n_scans,scan_points\n";
    header = true;
  }
  f << call << "," << total_ms << "," << h2d_ms << ","
    << kernel_ms << "," << d2h_ms << ","
    << n_total << "," << n_scans << "," << p << "\n";
}

void cpu_fallback_single(const std::vector<unsigned char>& grid,
                         const int w, const int h,
                         const std::vector<int>& px,
                         const std::vector<int>& py,
                         const std::vector<int>& cx,
                         const std::vector<int>& cy,
                         std::vector<float>* score) {
  const int n = (int)std::min(cx.size(), cy.size());
  const int p = (int)std::min(px.size(), py.size());
  score->assign(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px[j] + cx[i];
      const int y = py[j] + cy[i];
      if (x >= 0 && x < w && y >= 0 && y < h) sum += grid[y * w + x];
    }
    (*score)[i] = (float)sum / (255.0f * p);
  }
}

#ifdef __CUDACC__

// 배치 커널: 각 candidate가 scan_id로 자신의 px/py 구간을 찾아 계산
__global__ void score_all_batch_kernel(
    const unsigned char* __restrict__ grid, const int w, const int h,
    const int* __restrict__ all_px,
    const int* __restrict__ all_py,
    const int p,
    const int* __restrict__ cx,
    const int* __restrict__ cy,
    const int* __restrict__ scan_id,
    const int n,
    float* __restrict__ score) {

  __shared__ int warp_sums[4];

  const int cand = blockIdx.x;
  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int wid  = tid >> 5;
  if (cand >= n) return;

  const int sid      = __ldg(&scan_id[cand]);
  const int px_base  = sid * p;
  const int cxi      = __ldg(&cx[cand]);
  const int cyi      = __ldg(&cy[cand]);
  int local_sum = 0;

  for (int j = tid; j < p; j += blockDim.x) {
    const int x = __ldg(&all_px[px_base + j]) + cxi;
    const int y = __ldg(&all_py[px_base + j]) + cyi;
    if (x >= 0 && x < w && y >= 0 && y < h)
      local_sum += __ldg(&grid[y * w + x]);
  }

  // Warp Shuffle Reduction (PA01 gpu_final 방식)
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  if (lane == 0) warp_sums[wid] = local_sum;
  __syncthreads();

  if (wid == 0) {
    local_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0;
    for (int offset = 2; offset > 0; offset >>= 1)
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  if (tid == 0)
    score[cand] = (float)local_sum / (255.0f * p);
}

// CudaEvent 완전 제거: cudaMemcpy(non-async)가 이미 동기화 보장
struct BatchWorkspace {
  unsigned char* grid    = nullptr; size_t grid_cap    = 0;
  bool           grid_ready = false;
  int*           all_px  = nullptr; size_t all_px_cap  = 0;
  int*           all_py  = nullptr; size_t all_py_cap  = 0;
  int*           cx      = nullptr; size_t cx_cap      = 0;
  int*           cy      = nullptr; size_t cy_cap      = 0;
  int*           scan_id = nullptr; size_t scan_id_cap = 0;
  float*         score   = nullptr; size_t score_cap   = 0;
};

BatchWorkspace& get_batch_ws() { static BatchWorkspace g; return g; }

template<typename T>
bool grow_b(T** p, size_t* cap, size_t need) {
  if (*cap >= need) return true;
  if (*p) { cudaFree(*p); *p = nullptr; }
  bool ok = cudaMalloc(reinterpret_cast<void**>(p), need) == cudaSuccess;
  if (ok) *cap = need;
  return ok;
}

#endif  // __CUDACC__

}  // namespace

void make_cand(const int min_x, const int max_x, const int min_y,
               const int max_y, const int step,
               std::vector<int>* cx, std::vector<int>* cy) {
  if (!cx || !cy || step <= 0) return;
  for (int x = min_x; x <= max_x; x += step)
    for (int y = min_y; y <= max_y; y += step) {
      cx->push_back(x);
      cy->push_back(y);
    }
}

// 기존 score_all (Branch 등 소규모 호출 유지용)
void score_all(const std::vector<unsigned char>& grid, const int w,
               const int h, const std::vector<int>& px,
               const std::vector<int>& py, const std::vector<int>& cx,
               const std::vector<int>& cy, std::vector<float>* score) {
  if (!score) return;
  // Branch 호출 (cands=4): CPU fallback으로 처리 (GPU overhead 불필요)
  cpu_fallback_single(grid, w, h, px, py, cx, cy, score);
}

// 새 배치 함수: fast_matcher의 Score()에서 호출
// all_px/py: 모든 scan의 px/py를 이어붙인 배열 (n_scans × p 개)
// cx/cy: 모든 scan의 candidate를 이어붙인 배열
// scan_id: 각 candidate가 어느 scan에 속하는지
void score_all_batch(const std::vector<unsigned char>& grid,
                     const int w, const int h,
                     const std::vector<int>& all_px,
                     const std::vector<int>& all_py,
                     const int p,
                     const std::vector<int>& cx,
                     const std::vector<int>& cy,
                     const std::vector<int>& scan_id,
                     std::vector<float>* score) {
  if (!score) return;
  const int n = (int)std::min(cx.size(), cy.size());
  score->assign(n, 0.0f);
  if (n == 0 || p == 0) return;

  double total_ms = 0.0;
  float  h2d_ms = 0.f, kernel_ms = 0.f, d2h_ms = 0.f;
  bool   used_cuda = false;

#ifdef __CUDACC__
  {
    const size_t gb  = (size_t)w * h * sizeof(unsigned char);
    const size_t pxb = all_px.size() * sizeof(int);
    const size_t cb  = (size_t)n * sizeof(int);
    const size_t sb  = (size_t)n * sizeof(float);

    BatchWorkspace& g = get_batch_ws();
    bool ok = true;

    // Grid 상주 (처음 1회만)
    if (ok && !g.grid_ready) {
      ok = ok && grow_b(&g.grid, &g.grid_cap, gb);
      if (ok) {
        ok = cudaMemcpy(g.grid, grid.data(), gb, cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) g.grid_ready = true;
      }
    }

    ok = ok && grow_b(&g.all_px,  &g.all_px_cap,  pxb);
    ok = ok && grow_b(&g.all_py,  &g.all_py_cap,  pxb);
    ok = ok && grow_b(&g.cx,      &g.cx_cap,       cb);
    ok = ok && grow_b(&g.cy,      &g.cy_cap,       cb);
    ok = ok && grow_b(&g.scan_id, &g.scan_id_cap,  cb);
    ok = ok && grow_b(&g.score,   &g.score_cap,    sb);

    if (ok) {
      const auto t0 = std::chrono::high_resolution_clock::now();

      // H2D (cudaMemcpy는 동기적 → 완료까지 대기)
      ok = ok && cudaMemcpy(g.all_px,  all_px.data(),  pxb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.all_py,  all_py.data(),  pxb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.cx,      cx.data(),       cb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.cy,      cy.data(),       cb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.scan_id, scan_id.data(),  cb, cudaMemcpyHostToDevice)==cudaSuccess;

      if (ok) {
        constexpr int kTPB = 128;
        const size_t shmem = (kTPB / 32) * sizeof(int);
        cudaEvent_t ker_stop; cudaEventCreate(&ker_stop);
        score_all_batch_kernel<<<n, kTPB, shmem>>>(
            g.grid, w, h, g.all_px, g.all_py, p,
            g.cx, g.cy, g.scan_id, n, g.score);
        cudaEventRecord(ker_stop);
        cudaEventSynchronize(ker_stop);  // 커널 완료까지 효율적 대기
        cudaEventDestroy(ker_stop);
        ok = cudaGetLastError() == cudaSuccess;
      }

      // D2H
      if (ok)
        ok = cudaMemcpy(score->data(), g.score, sb, cudaMemcpyDeviceToHost)==cudaSuccess;

      if (ok) used_cuda = true;

      total_ms = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }
  }
  if (!used_cuda) {
    // CPU fallback: 개별 scan별로 처리
    for (int s = 0; s < (int)(all_px.size() / p); ++s) {
      const std::vector<int> px_s(all_px.begin() + s*p, all_px.begin() + (s+1)*p);
      const std::vector<int> py_s(all_py.begin() + s*p, all_py.begin() + (s+1)*p);
      std::vector<int> cx_s, cy_s;
      std::vector<size_t> idx;
      for (int i = 0; i < n; ++i)
        if (scan_id[i] == s) { cx_s.push_back(cx[i]); cy_s.push_back(cy[i]); idx.push_back(i); }
      std::vector<float> sc;
      cpu_fallback_single(grid, w, h, px_s, py_s, cx_s, cy_s, &sc);
      for (size_t k = 0; k < idx.size(); ++k) (*score)[idx[k]] = sc[k];
    }
  }
#else
  // No CUDA: CPU fallback
  for (int s = 0; s < (int)(all_px.size() / p); ++s) {
    const std::vector<int> px_s(all_px.begin() + s*p, all_px.begin() + (s+1)*p);
    const std::vector<int> py_s(all_py.begin() + s*p, all_py.begin() + (s+1)*p);
    std::vector<int> cx_s, cy_s; std::vector<size_t> idx;
    for (int i = 0; i < n; ++i)
      if (scan_id[i] == s) { cx_s.push_back(cx[i]); cy_s.push_back(cy[i]); idx.push_back(i); }
    std::vector<float> sc;
    cpu_fallback_single(grid, w, h, px_s, py_s, cx_s, cy_s, &sc);
    for (size_t k = 0; k < idx.size(); ++k) (*score)[idx[k]] = sc[k];
  }
#endif

  static int call_count = 0; ++call_count;
  const int n_scans = (int)(all_px.size() / p);
  std::fprintf(stderr,
    "[pa02_batch] call=%d total=%.3fms h2d=%.3fms kernel=%.3fms "
    "d2h=%.3fms cands=%d scans=%d cuda=%d\n",
    call_count, total_ms, h2d_ms, kernel_ms, d2h_ms,
    n, n_scans, used_cuda ? 1 : 0);
  std::fflush(stderr);
  WriteLog(call_count, n, n_scans, p, total_ms, h2d_ms, kernel_ms, d2h_ms);
}

}  // namespace cartographer_parallel
