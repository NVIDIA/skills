// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

/// Numerical comparison harness for Nsight AI Porting Agent standalone tests.
///
/// Copy this header into the test directory of the function under test and
/// include it. The gate is `wmad`, the normalized weighted mean absolute
/// difference against the reference field; `n_exceed` and the worst element
/// are reported so an outlier failure is reviewable rather than merely a
/// number to loosen. See references/verification-guide.md.
///
/// Every tolerance is read from the environment (WMAD_TOL, ABS_TOL,
/// REL_TOL, REL_FLOOR, MAX_REL_CHECK) so a test never hardcodes one.

#ifndef NSIGHT_AI_PORT_COMPARE_FIELD_HPP
#define NSIGHT_AI_PORT_COMPARE_FIELD_HPP

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace nsight_ai_port {

/// Outcome of one field comparison. `wmad` gates; the rest is diagnosis.
struct CompareResult {
  double wmad = 0.0;
  double max_rel = 0.0;
  long long n_exceed = 0;
  long long n_total = 0;
  std::size_t worst = 0;  ///< flat index, 0-based
  bool nonfinite = false;
  /// A zero-element comparison. Every metric is trivially 0 over an empty
  /// field, so it would otherwise pass while proving nothing.
  bool empty_field = false;
};

/// Gate thresholds. Defaults depend on the field's element type and are
/// then overridden from the environment.
struct Tolerances {
  double wmad = 1e-12;
  double abs_tol = 1e-12;
  double rel_tol = 1e-10;
  double rel_floor = 1e-10;
  bool max_rel_check = true;
};

inline double env_real(const char* name, double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw) return fallback;
  // Require the whole value, not just a numeric prefix: WMAD_TOL=1e-3oops
  // would otherwise silently become 1e-3 here while the Fortran harness
  // rejected it and kept its default, running the two stages of a
  // cross-language port at different tolerances.
  while (std::isspace(static_cast<unsigned char>(*end))) ++end;
  return (*end == '\0') ? parsed : fallback;
}

inline bool env_flag(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  // Lowercase first, so MAX_REL_CHECK=FALSE means the same thing here as it
  // does in the Fortran harness. The two must agree: the same environment
  // has to configure the same gate in either language.
  std::string v(raw);
  for (char& ch : v) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (v == "0" || v == "off" || v == "false" || v == "no") return false;
  if (v == "1" || v == "on" || v == "true" || v == "yes") return true;
  return fallback;
}

/// Kind-aware defaults, then environment overrides.
template <typename T>
inline Tolerances tolerances_for() {
  Tolerances tol;
  if (sizeof(T) <= sizeof(float)) {
    tol.wmad = 1e-6;
    tol.abs_tol = 1e-5;
    tol.rel_tol = 1e-4;
  }
  tol.wmad = env_real("WMAD_TOL", tol.wmad);
  tol.abs_tol = env_real("ABS_TOL", tol.abs_tol);
  tol.rel_tol = env_real("REL_TOL", tol.rel_tol);
  tol.rel_floor = env_real("REL_FLOOR", tol.rel_floor);
  tol.max_rel_check = env_flag("MAX_REL_CHECK", tol.max_rel_check);
  return tol;
}

/// Compare `a` against reference `ref` over `n` elements. `w` carries the
/// physical weight of each element (cell volume or area on a non-uniform
/// mesh) and may be nullptr, meaning 1. Accumulation is always in double,
/// whatever the field type.
template <typename T, typename W = T>
inline CompareResult compare_field(const T* a, const T* ref, std::size_t n,
                                   const Tolerances& tol,
                                   const W* w = nullptr) {
  CompareResult c;
  c.n_total = static_cast<long long>(n);
  if (n == 0) {
    c.empty_field = true;
    return c;
  }

  // Non-finite anywhere is an unconditional failure — never waivable.
  double peak = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(static_cast<double>(a[i])) ||
        !std::isfinite(static_cast<double>(ref[i]))) {
      c.nonfinite = true;
      c.worst = i;
      c.wmad = std::numeric_limits<double>::infinity();
      c.max_rel = std::numeric_limits<double>::infinity();
      c.n_exceed = -1;
      return c;
    }
    peak = std::fmax(peak, std::fabs(static_cast<double>(ref[i])));
  }

  // Magnitude floor: cells far below the field peak cannot manufacture a
  // huge relative error out of numerical dust.
  const double floor_v = tol.rel_floor * peak;
  double num = 0.0;
  double den = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const double wi = (w != nullptr) ? static_cast<double>(w[i]) : 1.0;
    const double ai = static_cast<double>(a[i]);
    const double ri = static_cast<double>(ref[i]);
    const double ad = std::fabs(ai - ri);
    num += wi * ad;
    den += wi * std::fabs(ri);
    const double rd = ad / (std::fabs(ri) + floor_v);
    if (rd > c.max_rel) {
      c.max_rel = rd;
      c.worst = i;
    }
    // An element passes if EITHER tolerance is met, so it counts as an
    // exceedance only when both are exceeded.
    if (ad > tol.abs_tol && rd > tol.rel_tol) ++c.n_exceed;
  }

  c.wmad = (den > 0.0) ? num / den : num;
  return c;
}

/// Print the four required numbers and apply the gate. Returns true when
/// the comparison passes; the caller decides how to fail (typically
/// `if (!report_and_gate(...)) return 1;`).
inline bool report_and_gate(const CompareResult& c, const Tolerances& tol,
                            const char* label) {
  if (c.empty_field) {
    std::printf("EMPTY FIELD: zero elements compared, so the result would "
                "prove nothing\n");
    std::printf("VERIFICATION FAILED (empty field): %s\n", label);
    return false;
  }

  if (c.nonfinite) {
    std::printf("NON-FINITE VALUES in %s at flat index %zu\n", label, c.worst);
    std::printf("VERIFICATION FAILED (non-finite): %s\n", label);
    return false;
  }

  std::printf("compare %s: wmad %.5e  max_rel %.5e\n", label, c.wmad,
              c.max_rel);
  std::printf("  n_exceed %lld of %lld  worst_idx %zu\n", c.n_exceed,
              c.n_total, c.worst);

  if (c.wmad > tol.wmad) {
    std::printf("VERIFICATION FAILED (wmad): %s\n", label);
    return false;
  }

  // The outlier bound IS reviewable — see verification-guide.md,
  // "Reviewing an outlier failure".
  if (tol.max_rel_check && c.n_exceed > 0) {
    std::printf("VERIFICATION FAILED (outliers) - review before waiving: %s\n",
                label);
    return false;
  }

  std::printf("VERIFICATION PASSED: %s\n", label);
  return true;
}

}  // namespace nsight_ai_port

#endif  // NSIGHT_AI_PORT_COMPARE_FIELD_HPP
