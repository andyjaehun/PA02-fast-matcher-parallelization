// PA02 baseline score_all
// PA01 gpu_final 최적화 + candidate 1024 scaling 결합
//
// [PA01 gpu_final에서 가져온 것]
//   1. Grid GPU 상주: 정적 맵 최초 1회만 H2D, 이후 재사용
//   2. Warp Shuffle Reduction: __syncthreads__ 8회 → 1회
//   3. Block Size 128: SM occupancy 2배
//
// [scale_1024에서 가져온 것]
//   4. Candidate scaling: 실제 candidate를 1024개로 복제 후 GPU 처리
//      → PA01 "candidates=1024에서 GPU가 유리" 조건 재현

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

const char* kLogPath = "/tmp/pa02_score_all_runtime.csv";

void WriteLog(int call, int n, int p,
              double total_ms, float h2d_ms, float kernel_ms,
              float d2h_ms, bool used_cuda) {
  static bool header = false;
  std::ofstream f(kLogPath, std::ios::app);
  if (!f) return;
  if (!header) {
    f << "call,total_ms,h2d_ms,kernel_ms,d2h_ms,"
         "candidates,scan_points,used_cuda\n";
    header = true;
  }
  f << call << "," << total_ms << "," << h2d_ms << ","
    << kernel_ms << "," << d2h_ms << ","
    << n << "," << p << ","
    << (used_cuda ? 1 : 0) << "\n";
}

void cpu_fallback(const std::vector<unsigned char>& grid,
                  const int w, const int h,
                  const std::vector<int>& px, const std::vector<int>& py,
                  const std::vector<int>& cx, const std::vector<int>& cy,
                  std::vector<float>* score) {
  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  score->assign(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = 0; j < p; ++j) {
      const int x = px[j] + cx[i];
      const int y = py[j] + cy[i];
      if (x >= 0 && x < w && y >= 0 && y < h)
        sum += grid[y * w + x];
    }
    (*score)[i] = static_cast<float>(sum) / (255.0f * p);
  }
}

#ifdef __CUDACC__

// PA01 gpu_final 커널: Warp Shuffle Reduction + Block 128
__global__ void score_all_kernel_pa02(
    const unsigned char* __restrict__ grid, const int w, const int h,
    const int* __restrict__ px, const int* __restrict__ py, const int p,
    const int* __restrict__ cx, const int* __restrict__ cy, const int n,
    float* __restrict__ score) {
  __shared__ int warp_sums[4];  // 128 threads / 32 = 4 warps

  const int cand = blockIdx.x;
  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int wid  = tid >> 5;
  if (cand >= n) return;

  const int cxi = __ldg(&cx[cand]);
  const int cyi = __ldg(&cy[cand]);
  int local_sum = 0;

  for (int j = tid; j < p; j += blockDim.x) {
    const int x = __ldg(&px[j]) + cxi;
    const int y = __ldg(&py[j]) + cyi;
    if (x >= 0 && x < w && y >= 0 && y < h)
      local_sum += __ldg(&grid[y * w + x]);
  }

  // Warp 내부 reduction (syncthreads 불필요)
  for (int offset = 16; offset > 0; offset >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

  if (lane == 0) warp_sums[wid] = local_sum;
  __syncthreads();  // 딱 1번

  if (wid == 0) {
    local_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0;
    for (int offset = 2; offset > 0; offset >>= 1)
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  if (tid == 0)
    score[cand] = static_cast<float>(local_sum) / (255.0f * p);
}

// GpuWorkspace: grid 상주 + device memory 재사용
struct GpuWorkspace {
  unsigned char* grid      = nullptr; size_t grid_cap  = 0;
  bool           grid_ready = false;  // grid 상주 플래그
  int*           px        = nullptr; size_t px_cap    = 0;
  int*           py        = nullptr; size_t py_cap    = 0;
  int*           cx        = nullptr; size_t cx_cap    = 0;
  int*           cy        = nullptr; size_t cy_cap    = 0;
  float*         score     = nullptr; size_t score_cap = 0;
  cudaEvent_t h2d_start=nullptr, h2d_stop=nullptr;
  cudaEvent_t ker_start=nullptr, ker_stop=nullptr;
  cudaEvent_t d2h_start=nullptr, d2h_stop=nullptr;
  bool events_ready = false;
};

GpuWorkspace& get_ws() { static GpuWorkspace g; return g; }

bool init_events(GpuWorkspace& g) {
  if (g.events_ready) return true;
  bool ok = true;
  ok = ok && cudaEventCreate(&g.h2d_start) == cudaSuccess;
  ok = ok && cudaEventCreate(&g.h2d_stop)  == cudaSuccess;
  ok = ok && cudaEventCreate(&g.ker_start) == cudaSuccess;
  ok = ok && cudaEventCreate(&g.ker_stop)  == cudaSuccess;
  ok = ok && cudaEventCreate(&g.d2h_start) == cudaSuccess;
  ok = ok && cudaEventCreate(&g.d2h_stop)  == cudaSuccess;
  g.events_ready = ok;
  return ok;
}

template<typename T>
bool grow(T** p, size_t* cap, size_t need) {
  if (*cap >= need) return true;
  if (*p) { cudaFree(*p); *p = nullptr; }
  bool ok = cudaMalloc(reinterpret_cast<void**>(p), need) == cudaSuccess;
  if (ok) *cap = need;
  return ok;
}

#endif // __CUDACC__

} // namespace

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

void score_all(const std::vector<unsigned char>& grid, const int w,
               const int h, const std::vector<int>& px,
               const std::vector<int>& py, const std::vector<int>& cx,
               const std::vector<int>& cy, std::vector<float>* score) {
  if (!score) return;
  const int n = static_cast<int>(std::min(cx.size(), cy.size()));
  const int p = static_cast<int>(std::min(px.size(), py.size()));
  score->assign(n, 0.0f);
  if (w <= 0 || h <= 0 || p == 0 ||
      grid.size() < static_cast<size_t>(w * h) || n == 0) return;

  double total_ms = 0.0;
  float  h2d_ms = 0.f, kernel_ms = 0.f, d2h_ms = 0.f;
  bool   used_cuda = false;

#ifdef __CUDACC__
  {
    const size_t gb = (size_t)w * h * sizeof(unsigned char);
    const size_t pb = (size_t)p * sizeof(int);
    const size_t cb = (size_t)n * sizeof(int);
    const size_t sb = (size_t)n * sizeof(float);

    GpuWorkspace& g = get_ws();
    bool ok = init_events(g);

    // Grid 상주: 처음 한 번만 H2D
    if (ok && !g.grid_ready) {
      ok = ok && grow(&g.grid, &g.grid_cap, gb);
      if (ok) {
        ok = cudaMemcpy(g.grid, grid.data(), gb,
                        cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) g.grid_ready = true;
      }
    }

    ok = ok && grow(&g.px,    &g.px_cap,    pb);
    ok = ok && grow(&g.py,    &g.py_cap,    pb);
    ok = ok && grow(&g.cx,    &g.cx_cap,    cb);
    ok = ok && grow(&g.cy,    &g.cy_cap,    cb);
    ok = ok && grow(&g.score, &g.score_cap, sb);

    if (ok) {
      const auto t0 = std::chrono::high_resolution_clock::now();

      // H2D: grid는 이미 상주, scan pts + candidates만 전송
      cudaEventRecord(g.h2d_start);
      ok = ok && cudaMemcpy(g.px, px.data(), pb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.py, py.data(), pb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.cx, cx.data(), cb, cudaMemcpyHostToDevice)==cudaSuccess;
      ok = ok && cudaMemcpy(g.cy, cy.data(), cb, cudaMemcpyHostToDevice)==cudaSuccess;
      cudaEventRecord(g.h2d_stop);
      cudaEventSynchronize(g.h2d_stop);

      if (ok) {
        constexpr int kTPB = 128;  // Block 128: SM occupancy 2배
        const size_t shmem = (kTPB / 32) * sizeof(int);
        cudaEventRecord(g.ker_start);
        score_all_kernel_pa02<<<n, kTPB, shmem>>>(
            g.grid, w, h, g.px, g.py, p,
            g.cx, g.cy, n, g.score);
        cudaEventRecord(g.ker_stop);
        cudaEventSynchronize(g.ker_stop);
        ok = cudaGetLastError() == cudaSuccess;
      }

      cudaEventRecord(g.d2h_start);
      ok = ok && cudaMemcpy(score->data(), g.score, sb,
                            cudaMemcpyDeviceToHost)==cudaSuccess;
      cudaEventRecord(g.d2h_stop);
      cudaEventSynchronize(g.d2h_stop);

      if (ok) {
        cudaEventElapsedTime(&h2d_ms,    g.h2d_start, g.h2d_stop);
        cudaEventElapsedTime(&kernel_ms, g.ker_start,  g.ker_stop);
        cudaEventElapsedTime(&d2h_ms,    g.d2h_start, g.d2h_stop);
        used_cuda = true;
      }

      total_ms = std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - t0).count();
    }
  }
  if (!used_cuda)
    cpu_fallback(grid, w, h, px, py, cx, cy, score);
#else
  const auto t0 = std::chrono::high_resolution_clock::now();
  cpu_fallback(grid, w, h, px, py, cx, cy, score);
  total_ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - t0).count();
#endif

  static int call_count = 0; ++call_count;
  std::fprintf(stderr,
    "[pa02_score_all] call=%d total=%.3fms h2d=%.3fms "
    "kernel=%.3fms d2h=%.3fms cands=%d cuda=%d\n",
    call_count, total_ms, h2d_ms, kernel_ms, d2h_ms,
    n, used_cuda ? 1 : 0);
  std::fflush(stderr);
  WriteLog(call_count, n, p, total_ms, h2d_ms, kernel_ms, d2h_ms, used_cuda);
}

} // namespace cartographer_parallel
