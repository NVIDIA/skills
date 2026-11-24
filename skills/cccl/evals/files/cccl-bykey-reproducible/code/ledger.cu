// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// End-of-day ledger: total traded amount per account. Trades arrive sorted by account.
// Build: nvcc -O2 -std=c++17 -arch=native ledger.cu -o ledger

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int num_accounts = 160;
constexpr int num_trades   = 16'000'000;

// Per-account totals in float. Returns the number of accounts.
int account_totals(const thrust::device_vector<int>& account,
                   const thrust::device_vector<float>& amount,
                   thrust::device_vector<int>& out_account,
                   thrust::device_vector<float>& out_total)
{
  auto end =
    thrust::reduce_by_key(account.begin(), account.end(), amount.begin(), out_account.begin(), out_total.begin());
  return static_cast<int>(end.first - out_account.begin());
}

int main()
{
  // Deterministic synthetic input: account sizes vary, amounts span several orders of magnitude.
  std::vector<int> h_account(num_trades);
  std::vector<float> h_amount(num_trades);
  std::vector<double> ref(num_accounts, 0.0);
  unsigned s = 12345u;
  for (int i = 0; i < num_trades; ++i)
  {
    s            = s * 1664525u + 1013904223u;
    h_account[i] = static_cast<int>((static_cast<long long>(i) * num_accounts) / num_trades);
    h_amount[i]  = static_cast<float>((s >> 8) % 100000) * 0.01f + static_cast<float>((s >> 28)) * 1000.0f;
    ref[h_account[i]] += h_amount[i];
  }
  thrust::device_vector<int> account  = h_account;
  thrust::device_vector<float> amount = h_amount;
  thrust::device_vector<int> out_account(num_accounts);
  thrust::device_vector<float> out_total(num_accounts);

  int n                            = account_totals(account, amount, out_account, out_total); // warm-up
  thrust::host_vector<float> first = out_total;

  int differing = 0;
  double ms     = 0;
  for (int run = 0; run < 20; ++run)
  {
    auto t0 = std::chrono::steady_clock::now();
    n       = account_totals(account, amount, out_account, out_total);
    cudaDeviceSynchronize();
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    thrust::host_vector<float> again = out_total;
    differing += std::memcmp(first.data(), again.data(), n * sizeof(float)) != 0;
  }

  double max_rel = 0;
  for (int a = 0; a < n; ++a)
  {
    double rel = std::abs(first[a] - ref[a]) / ref[a];
    max_rel    = rel > max_rel ? rel : max_rel;
  }
  std::printf("accounts: %d\n", n);
  std::printf("reruns that differ bitwise: %d of 20\n", differing);
  std::printf("max relative error vs double reference: %.2e\n", max_rel);
  std::printf("time per call: %.3f ms\n", ms / 20);
  return 0;
}
