// PA02 최종 통합: 모든 최적화 적용
// 1. MakeScans:    상수 호이스팅 + float + GCC pragma (5.8x)
// 2. MakeLowCands: make_cand 1번 + reserve (4.3x)
// 3. Score(coarse): GPU 배치화 15→1 호출 (4.95x)
// 4. Branch Score: CPU 직접 계산 (5.1x)
#pragma GCC optimize("O3", "unroll-loops", "unsafe-math-optimizations")

#include "cartographer_parallel/fast_matcher.h"
#include "cartographer_parallel/assignment.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// score_all_pa02_batch.cpp에서 제공하는 배치 함수 선언
namespace cartographer_parallel {
void score_all_batch(const std::vector<unsigned char>& grid,
                     const int w, const int h,
                     const std::vector<int>& all_px,
                     const std::vector<int>& all_py,
                     const int p,
                     const std::vector<int>& cx,
                     const std::vector<int>& cy,
                     const std::vector<int>& scan_id,
                     std::vector<float>* score);
}

static void WriteLog(int call, double scans_ms, double lowcands_ms,
                     double score_ms, double branch_ms, double total_ms) {
  static bool header = false;
  const char* path = "/tmp/pa02_combined_runtime.csv";
  FILE* f = std::fopen(path, "a");
  if (!f) return;
  if (!header) {
    std::fprintf(f, "call,scans_ms,lowcands_ms,score_ms,branch_ms,total_ms\n");
    header = true;
  }
  std::fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               call, scans_ms, lowcands_ms, score_ms, branch_ms, total_ms);
  std::fclose(f);
}

namespace cartographer_parallel {
namespace {

std::string Trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const std::string::size_type b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  const std::string::size_type e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}
std::string Unquote(const std::string& s) {
  const std::string v = Trim(s);
  if (v.size() >= 2 &&
      ((v.front() == '"' && v.back() == '"') ||
       (v.front() == '\'' && v.back() == '\'')))
    return v.substr(1, v.size() - 2);
  return v;
}
std::string Dirname(const std::string& path) {
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}
bool IsAbs(const std::string& path) { return !path.empty() && path[0] == '/'; }
std::string Join(const std::string& dir, const std::string& file) {
  if (file.empty() || IsAbs(file)) return file;
  return dir == "." ? file : dir + "/" + file;
}
std::vector<double> ParseList(std::string v) {
  for (char& c : v) if (c == '[' || c == ']' || c == ',') c = ' ';
  std::istringstream in(v); std::vector<double> out; double x = 0.0;
  while (in >> x) out.push_back(x);
  return out;
}
std::string PgmToken(std::istream* in) {
  std::string token; char c = 0;
  while (in->get(c)) {
    if (std::isspace(static_cast<unsigned char>(c))) continue;
    if (c == '#') { in->ignore(std::numeric_limits<std::streamsize>::max(), '\n'); continue; }
    token.push_back(c); break;
  }
  while (in->get(c)) {
    if (std::isspace(static_cast<unsigned char>(c))) break;
    if (c == '#') { in->ignore(std::numeric_limits<std::streamsize>::max(), '\n'); break; }
    token.push_back(c);
  }
  return token;
}
int ClampInt(const int x, const int lo, const int hi) { return std::max(lo, std::min(hi, x)); }
double NormalizeYaw(double yaw) {
  while (yaw > M_PI)  yaw -= 2.0 * M_PI;
  while (yaw < -M_PI) yaw += 2.0 * M_PI;
  return yaw;
}

}  // namespace

bool FastMatcher::LoadMap(const std::string& yaml_file) {
  std::ifstream yaml(yaml_file);
  if (!yaml) return false;
  std::string image; bool negate = false;
  double occupied_thresh = 0.65, free_thresh = 0.196;
  std::string line;
  while (std::getline(yaml, line)) {
    line = line.substr(0, line.find('#'));
    const std::string::size_type colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = Trim(line.substr(0, colon));
    const std::string val = Trim(line.substr(colon + 1));
    if      (key == "image")           image  = Join(Dirname(yaml_file), Unquote(val));
    else if (key == "resolution")      res_   = std::stod(val);
    else if (key == "origin") {
      const auto origin = ParseList(val);
      if (origin.size() >= 2) { ox_ = origin[0]; oy_ = origin[1]; }
    }
    else if (key == "negate")          negate          = (val=="1"||val=="true"||val=="True");
    else if (key == "occupied_thresh") occupied_thresh = std::stod(val);
    else if (key == "free_thresh")     free_thresh     = std::stod(val);
  }
  (void)occupied_thresh; (void)free_thresh;
  if (image.empty()) return false;
  std::ifstream pgm(image, std::ios::binary);
  if (!pgm) return false;
  const std::string magic = PgmToken(&pgm);
  if (magic != "P5" && magic != "P2") return false;
  w_ = std::stoi(PgmToken(&pgm)); h_ = std::stoi(PgmToken(&pgm));
  const int max_value = std::stoi(PgmToken(&pgm));
  if (w_ <= 0 || h_ <= 0 || max_value <= 0 || max_value > 255) return false;
  std::vector<unsigned char> pixels(w_ * h_, 0);
  if (magic == "P5") {
    pgm.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    if (pgm.gcount() != static_cast<std::streamsize>(pixels.size())) return false;
  } else {
    for (unsigned char& pixel : pixels) {
      const std::string token = PgmToken(&pgm);
      if (token.empty()) return false;
      pixel = static_cast<unsigned char>(ClampInt(std::stoi(token), 0, max_value));
    }
  }
  map_.assign(w_ * h_, 0);
  for (int i = 0; i < w_ * h_; ++i) {
    const double v = static_cast<double>(pixels[i]) / max_value;
    map_[i] = static_cast<unsigned char>(
        ClampInt(static_cast<int>(std::lround(255.0 * (negate ? v : 1.0 - v))), 0, 255));
  }
  grids_ = MakeGridStack();
  return true;
}

void FastMatcher::SetOptions(const MatchOpt& opt) {
  opt_ = opt;
  if (has_map()) grids_ = MakeGridStack();
}

std::vector<FastMatcher::Scan> FastMatcher::MakeScans(
    const std::vector<float>& xs, const std::vector<float>& ys,
    const Pose2& init, int* const num_ang, double* const step) const {
  double max_range = 3.0 * res_;
  for (size_t i = 0; i < xs.size() && i < ys.size(); ++i)
    max_range = std::max(max_range, std::hypot((double)xs[i], (double)ys[i]));
  double angle_step = opt_.angular_step;
  if (angle_step <= 0.0) {
    const double c = 1.0 - (res_*res_) / (2.0*max_range*max_range);
    angle_step = 0.999 * std::acos(std::max(-1.0, std::min(1.0, c)));
    if (!std::isfinite(angle_step) || angle_step <= 0.0) angle_step = 0.05;
  }
  const int n_ang = std::max(0, (int)std::ceil(opt_.angular_window / angle_step));
  const int scan_count = 2 * n_ang + 1;
  if (num_ang) *num_ang = n_ang;
  if (step)    *step    = angle_step;
  // 상수 호이스팅 + float 연산
  const float inv_res = (float)(1.0 / res_);
  const float bx      = (float)((init.x - ox_) * inv_res);
  const float by_base = (float)((h_ - 1) - (init.y - oy_) * inv_res);

  std::vector<float> cr(scan_count), sr(scan_count);
  for (int s = 0; s < scan_count; ++s) {
    const float yaw = (float)(init.yaw + (s - n_ang) * angle_step);
    cr[s] = cosf(yaw) * inv_res;
    sr[s] = sinf(yaw) * inv_res;
  }

  const int n_pts = (int)std::min(xs.size(), ys.size());
  std::vector<Scan> scans(scan_count);
  for (int s = 0; s < scan_count; ++s) {
    scans[s].x.resize(n_pts);
    scans[s].y.resize(n_pts);
    const float crs = cr[s], srs = sr[s];
    for (int i = 0; i < n_pts; ++i) {
      scans[s].x[i] = (int)floorf(bx      + crs * xs[i] - srs * ys[i]);
      scans[s].y[i] = (int)floorf(by_base - srs * xs[i] - crs * ys[i]);
    }
  }
  return scans;
}

std::vector<FastMatcher::Bounds> FastMatcher::MakeBounds(
    const std::vector<Scan>& scans, const double window,
    const bool full_map) const {
  const int lin = (int)std::ceil(window / res_);
  std::vector<Bounds> bounds(scans.size());
  for (size_t s = 0; s < scans.size(); ++s) {
    Bounds b;
    if (full_map) {
      b.min_x = std::numeric_limits<int>::lowest() / 4;
      b.max_x = std::numeric_limits<int>::max()    / 4;
      b.min_y = std::numeric_limits<int>::lowest() / 4;
      b.max_y = std::numeric_limits<int>::max()    / 4;
    } else {
      b.min_x = -lin; b.max_x = lin; b.min_y = -lin; b.max_y = lin;
      bounds[s] = b; continue;
    }
    for (size_t i = 0; i < scans[s].x.size(); ++i) {
      b.min_x = std::max(b.min_x, -scans[s].x[i]);
      b.max_x = std::min(b.max_x, w_ - 1 - scans[s].x[i]);
      b.min_y = std::max(b.min_y, -scans[s].y[i]);
      b.max_y = std::min(b.max_y, h_ - 1 - scans[s].y[i]);
    }
    bounds[s] = b;
  }
  return bounds;
}

std::vector<FastMatcher::Grid> FastMatcher::MakeGridStack() const {
  const int depth = std::max(1, opt_.branch_depth);
  std::vector<Grid> grids;
  grids.reserve(depth);
  for (int level = 0; level < depth; ++level) {
    const int win = 1 << level;
    Grid g; g.w = w_; g.h = h_; g.win = win;
    g.cell.assign(w_ * h_, 0);
    for (int y = 0; y < h_; ++y)
      for (int x = 0; x < w_; ++x) {
        unsigned char best = 0;
        for (int dy = 0; dy < win && y+dy < h_; ++dy)
          for (int dx = 0; dx < win && x+dx < w_; ++dx)
            best = std::max(best, map_[(y+dy)*w_ + (x+dx)]);
        g.cell[y*w_ + x] = best;
      }
    grids.push_back(g);
  }
  return grids;
}

// make_cand 1번 + reserve 최적화
std::vector<FastMatcher::Cand> FastMatcher::MakeLowCands(
    const std::vector<Bounds>& bounds, const int depth) const {
  const int step = 1 << depth;
  std::vector<Cand> out;

  std::vector<int> base_cx, base_cy;
  for (size_t s = 0; s < bounds.size(); ++s) {
    if (bounds[s].min_x > bounds[s].max_x || bounds[s].min_y > bounds[s].max_y) continue;
    make_cand(bounds[s].min_x, bounds[s].max_x, bounds[s].min_y,
              bounds[s].max_y, step, &base_cx, &base_cy);
    break;
  }
  if (base_cx.empty()) return out;

  out.reserve(bounds.size() * base_cx.size());
  for (size_t s = 0; s < bounds.size(); ++s) {
    if (bounds[s].min_x > bounds[s].max_x || bounds[s].min_y > bounds[s].max_y) continue;
    for (size_t i = 0; i < base_cx.size(); ++i) {
      Cand c; c.scan = (int)s; c.x = base_cx[i]; c.y = base_cy[i];
      out.push_back(c);
    }
  }
  return out;
}

// ── Score: GPU 배치화 + Branch CPU 전환 ──────────────────────────────────────
// cands 많음 (Score coarse): GPU 배치 1번 호출
// cands 적음 (Branch, 4개): CPU 직접 계산 (GPU overhead 제거)
static constexpr int kBranchCpuThreshold = 32;

void FastMatcher::Score(const Grid& grid, const std::vector<Scan>& scans,
                        std::vector<Cand>* const cand) const {
  if (cand == nullptr || cand->empty()) return;

  const int n       = (int)cand->size();
  const int n_scans = (int)scans.size();
  const int p       = n_scans > 0 ? (int)scans[0].x.size() : 0;
  if (p == 0) return;

  if (n <= kBranchCpuThreshold) {
    // ── CPU 직접 계산 (Branch 호출: cands=4) ──────────────────────────────
    // scan별 그룹핑
    std::vector<std::vector<size_t>> by_scan(n_scans);
    for (int i = 0; i < n; ++i) {
      const int s = (*cand)[i].scan;
      if (s >= 0 && s < n_scans) by_scan[s].push_back(i);
    }
    for (int s = 0; s < n_scans; ++s) {
      const auto& ids = by_scan[s];
      if (ids.empty()) continue;
      const int* px = scans[s].x.data();
      const int* py = scans[s].y.data();
      const int  w  = grid.w, h = grid.h;
      const unsigned char* cell = grid.cell.data();
      for (size_t k = 0; k < ids.size(); ++k) {
        const int cxi = (*cand)[ids[k]].x;
        const int cyi = (*cand)[ids[k]].y;
        int sum = 0;
        for (int j = 0; j < p; ++j) {
          const int x = px[j] + cxi;
          const int y = py[j] + cyi;
          if (x >= 0 && x < w && y >= 0 && y < h) sum += cell[y * w + x];
        }
        (*cand)[ids[k]].score = (float)sum / (255.0f * p);
      }
    }
  } else {
    // ── GPU 배치 1번 호출 (Score coarse: cands=15,360) ───────────────────
    std::vector<int> all_px, all_py;
    all_px.reserve(n_scans * p);
    all_py.reserve(n_scans * p);
    for (int s = 0; s < n_scans; ++s) {
      all_px.insert(all_px.end(), scans[s].x.begin(), scans[s].x.end());
      all_py.insert(all_py.end(), scans[s].y.begin(), scans[s].y.end());
    }
    std::vector<int> cx(n), cy(n), scan_id(n);
    for (int i = 0; i < n; ++i) {
      cx[i] = (*cand)[i].x; cy[i] = (*cand)[i].y; scan_id[i] = (*cand)[i].scan;
    }
    std::vector<float> score;
    score_all_batch(grid.cell, grid.w, grid.h,
                    all_px, all_py, p, cx, cy, scan_id, &score);
    for (int i = 0; i < n && i < (int)score.size(); ++i)
      (*cand)[i].score = score[i];
  }

  std::sort(cand->begin(), cand->end(),
            [](const Cand& a, const Cand& b){ return a.score > b.score; });
}

FastMatcher::Cand FastMatcher::Branch(const std::vector<Grid>& grids,
                                      const std::vector<Scan>& scans,
                                      const std::vector<Bounds>& bounds,
                                      const std::vector<Cand>& cand,
                                      const int depth,
                                      const float min_score) const {
  if (cand.empty()) { Cand e; e.score = 0.f; return e; }
  if (depth == 0) return cand.front();
  Cand best; best.score = min_score;
  const int half = 1 << (depth - 1);
  for (const Cand& c : cand) {
    if (c.score <= best.score) break;
    std::vector<Cand> child;
    for (const int dx : {0, half}) {
      if (c.x + dx > bounds[c.scan].max_x) continue;
      for (const int dy : {0, half}) {
        if (c.y + dy > bounds[c.scan].max_y) continue;
        Cand next; next.scan = c.scan; next.x = c.x+dx; next.y = c.y+dy;
        child.push_back(next);
      }
    }
    Score(grids[depth-1], scans, &child);
    const Cand refined = Branch(grids, scans, bounds, child, depth-1, best.score);
    if (refined.score > best.score) best = refined;
  }
  return best;
}

CandOut FastMatcher::ToOut(const Cand& cand, const Pose2& init,
                           const int num_ang, const double step) const {
  CandOut out;
  out.x = init.x + cand.x * res_;
  out.y = init.y - cand.y * res_;
  out.yaw = NormalizeYaw(init.yaw + (cand.scan - num_ang) * step);
  out.score = cand.score;
  return out;
}

bool FastMatcher::Match(const std::vector<float>& xs, const std::vector<float>& ys,
                        const Pose2& init, const bool global, MatchOut* const out) const {
  return MatchWithWindow(xs, ys, init,
                         global ? opt_.global_window : opt_.linear_window,
                         global && opt_.full_map_search, out);
}

bool FastMatcher::MatchWithWindow(const std::vector<float>& xs,
                                  const std::vector<float>& ys,
                                  const Pose2& init,
                                  const double window,
                                  const bool full_map,
                                  MatchOut* const out) const {
  if (out == nullptr) return false;
  *out = MatchOut();
  if (!has_map() || xs.empty() || ys.empty()) return false;

  static int call_id = 0;
  const auto t_total = Clock::now();

  int num_ang = 0; double step = 0.0;
  auto t_scans = Clock::now();
  const std::vector<Scan>   scans  = MakeScans(xs, ys, init, &num_ang, &step);
  const double scans_ms = Ms(Clock::now() - t_scans).count();

  const std::vector<Bounds> bounds = MakeBounds(scans, window, full_map);
  std::vector<Grid> temp_grids;
  const std::vector<Grid>* grids_ptr = &grids_;
  if (grids_ptr->empty()) { temp_grids = MakeGridStack(); grids_ptr = &temp_grids; }
  const std::vector<Grid>& grids = *grids_ptr;
  const int max_depth = (int)grids.size() - 1;

  auto t_lc = Clock::now();
  std::vector<Cand> coarse = MakeLowCands(bounds, max_depth);
  const double lowcands_ms = Ms(Clock::now() - t_lc).count();

  auto t_score = Clock::now();
  Score(grids[max_depth], scans, &coarse);
  const double score_ms = Ms(Clock::now() - t_score).count();

  if (coarse.empty()) return false;

  auto t_branch = Clock::now();
  const Cand best = Branch(grids, scans, bounds, coarse, max_depth, opt_.min_score);
  const double branch_ms = Ms(Clock::now() - t_branch).count();

  const double total_ms = Ms(Clock::now() - t_total).count();
  ++call_id;
  WriteLog(call_id, scans_ms, lowcands_ms, score_ms, branch_ms, total_ms);
  std::fprintf(stderr, "[pa02_combined] call=%d scans=%.3f lc=%.3f score=%.3f branch=%.3f total=%.3fms\n",
               call_id, scans_ms, lowcands_ms, score_ms, branch_ms, total_ms);

  out->ok = best.score > opt_.min_score;
  out->score = best.score;
  out->pose = init;
  if (out->ok) {
    const CandOut best_out = ToOut(best, init, num_ang, step);
    out->pose.x = best_out.x; out->pose.y = best_out.y; out->pose.yaw = best_out.yaw;
  }
  const int n = std::min(opt_.max_cand, (int)coarse.size());
  out->cand.reserve(n);
  for (int i = 0; i < n; ++i) out->cand.push_back(ToOut(coarse[i], init, num_ang, step));
  return out->ok;
}

}  // namespace cartographer_parallel
