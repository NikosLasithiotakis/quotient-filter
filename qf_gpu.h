#ifndef QF_GPU_H
#define QF_GPU_H

#include "qf.h"

#ifdef __cplusplus
extern "C" {
#endif

struct quotient_filter *qf_copy_to_gpu(struct quotient_filter *cpu_qf);

void qf_free_gpu(struct quotient_filter *d_qf);

bool qf_gpu_lookup(struct quotient_filter *d_qf, uint64_t hash);

#ifdef __cplusplus
}
#endif

#endif // QF_GPU_H
