#include "model.h"
#include <omp.h>
API int qf_threads() { return omp_get_num_procs(); }
API int qf_simulate(const Parameters* p, std::uint64_t start, std::uint64_t count,
                    int threads, int stride, double* output) {
    if (!p || !output || p->assets < 1 || p->assets > MAX_ASSETS || p->steps < 1 ||
        threads < 1 || (stride != 1 && stride != p->assets + 1)) return -1;
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i)
        scenario(*p, start + i, output + i * stride, stride);
    return 0;
}
