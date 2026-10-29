#include "model.h"
#include <algorithm>
#include <omp.h>

// Paired moving-block bootstrap preserves dependence within sampled blocks.
API int qf_bootstrap(const double* returns, int rows, int assets, const int* pairs,
                     int pair_count, int repetitions, int block, std::uint64_t seed,
                     int threads, double* output) {
    if (!returns || !pairs || !output || rows < 2 || assets < 2 || pair_count < 1 ||
        repetitions < 1 || block < 1 || threads < 1) return -1;
    for (int p = 0; p < pair_count * 2; ++p)
        if (pairs[p] < 0 || pairs[p] >= assets) return -2;
    omp_set_dynamic(0);
    std::int64_t jobs = std::int64_t(pair_count) * repetitions;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t job = 0; job < jobs; ++job) {
        int pair = int(job / repetitions), a = pairs[2 * pair], b = pairs[2 * pair + 1];
        std::uint64_t key = mix(seed ^ mix(std::uint64_t(job)));
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        int drawn = 0, draw = 0;
        while (drawn < rows) {
            int start = int(mix(key + draw++) % std::uint64_t(rows));
            for (int k = 0; k < block && drawn < rows; ++k, ++drawn) {
                auto index = std::int64_t((start + k) % rows) * assets;
                double x = returns[index + a], y = returns[index + b];
                sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y;
            }
        }
        double denominator = sqrt(fmax(0.0, (sxx-sx*sx/rows)*(syy-sy*sy/rows)));
        output[job] = denominator > 0 ? std::clamp((sxy-sx*sy/rows)/denominator, -1.0, 1.0) : NAN;
    }
    return 0;
}
