#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
constexpr int MAX_ASSETS = 16;
struct Parameters {
    int assets, steps;
    double horizon;
    std::uint64_t seed;
    double weights[MAX_ASSETS], mu[MAX_ASSETS], sigma[MAX_ASSETS];
    double chol[MAX_ASSETS * MAX_ASSETS];
};
// SplitMix64 counter permutation. Streams depend on seed, global path and draw,
// never on thread, batch boundaries, launch geometry or scheduling.
HD inline std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
HD inline double uniform(std::uint64_t x) {
    return (double(mix(x) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
HD inline double normal(std::uint64_t key, std::uint64_t draw) {
    double u = uniform(key + draw * 2), v = uniform(key + draw * 2 + 1);
    return sqrt(-2.0 * log(u)) * cos(6.2831853071795864769 * v);
}
HD inline void scenario(const Parameters& p, std::uint64_t id, double* out, int stride) {
    double logs[MAX_ASSETS] = {}, z[MAX_ASSETS];
    double dt = p.horizon / p.steps, sqdt = sqrt(dt);
    auto key = mix(id) ^ mix(p.seed);
    for (int s = 0; s < p.steps; ++s) {
        for (int a = 0; a < p.assets; ++a)
            z[a] = normal(key, std::uint64_t(s) * p.assets + a);
        for (int a = 0; a < p.assets; ++a) {
            double shock = 0;
            for (int b = 0; b <= a; ++b) shock += p.chol[a * MAX_ASSETS + b] * z[b];
            logs[a] += (p.mu[a] - 0.5 * p.sigma[a] * p.sigma[a]) * dt + p.sigma[a] * sqdt * shock;
        }
    }
    double portfolio = 0;
    for (int a = 0; a < p.assets; ++a) {
        double value = exp(logs[a]);
        portfolio += p.weights[a] * value;
        if (stride > 1) out[a + 1] = value;
    }
    out[0] = portfolio;
}
