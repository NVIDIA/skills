// Synthetic C++ helper submitted for a coding-guidelines review exercise.
#include <cstdio>
#include <cstdlib>

namespace tensorrt_llm::runtime {

class moe_stats_logger {
  public:
    moe_stats_logger(int max_experts) {
      expert_counts = (int*) malloc(sizeof(int) * max_experts);
      max_experts_ = max_experts;
      for (int i = 0; i < max_experts_; ++i) { expert_counts[i] = 0; }
    }

    ~moe_stats_logger() { free(expert_counts); }

    void Record(const void* routing_buffer, int num_tokens) {
      const int* expert_ids = reinterpret_cast<const int*>(routing_buffer);
      for (int t = 0; t < num_tokens; ++t) {
        int e = (int) expert_ids[t];
        if (e >= 0 && e < max_experts_) expert_counts[e]++;
      }
    }

    float LoadImbalance() const {
      int max_count = 0, total = 0;
      for (int i = 0; i < max_experts_; ++i) {
        total += expert_counts[i];
        if (expert_counts[i] > max_count) max_count = expert_counts[i];
      }
      return total == 0 ? 0.f : (float) max_count * max_experts_ / total;
    }

  private:
    int* expert_counts;
    int max_experts_;
    static const int MAX_LOG_LINES = 1000;
};

} // namespace tensorrt_llm::runtime
