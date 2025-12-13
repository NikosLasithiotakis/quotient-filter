#include "qf.h"
#include "qf_gpu.h"

#define LOW_MASK(n) ((1ULL << (n)) - 1ULL)

__device__ __forceinline__ int is_occupied(uint64_t elt)
{
	return elt & 1;
}

__device__ __forceinline__ int is_continuation(uint64_t elt)
{
	return elt & 2;
}

__device__ __forceinline__ int is_shifted(uint64_t elt)
{
	return elt & 4;
}

__device__ __forceinline__ uint64_t get_remainder(uint64_t elt)
{
	return elt >> 3;
}

__device__ __forceinline__ uint64_t hash_to_quotient(const struct quotient_filter *qf, uint64_t hash)
{
	return (hash >> qf->qf_rbits) & qf->qf_index_mask;
}

__device__ __forceinline__ uint64_t hash_to_remainder(const struct quotient_filter *qf, uint64_t hash)
{
	return hash & qf->qf_rmask;
}

__device__ __forceinline__ uint64_t incr(const struct quotient_filter *qf, uint64_t idx)
{
	return (idx + 1) & qf->qf_index_mask;
}

__device__ __forceinline__ uint64_t decr(const struct quotient_filter *qf, uint64_t idx)
{
	return (idx - 1) & qf->qf_index_mask;
}

__device__ uint64_t get_elem(const struct quotient_filter *qf, uint64_t idx)
{
	uint64_t elt = 0;
	size_t bitpos = qf->qf_elem_bits * idx;
	size_t tabpos = bitpos / 64;
	size_t slotpos = bitpos % 64;
	int spillbits = (slotpos + qf->qf_elem_bits) - 64;

	elt = (qf->qf_table[tabpos] >> slotpos) & qf->qf_elem_mask;

	if (spillbits > 0) {
		++tabpos;
		uint64_t x = qf->qf_table[tabpos] & LOW_MASK(spillbits);
		elt |= x << (qf->qf_elem_bits - spillbits);
	}
	return elt;
}

__device__ uint64_t find_run_index(const struct quotient_filter *qf, uint64_t fq)
{
	uint64_t b = fq;
	while (is_shifted(get_elem(qf, b))) {
		b = decr(qf, b);
	}

	uint64_t s = b;
	while (b != fq) {
		do {
			s = incr(qf, s);
		} while (is_continuation(get_elem(qf, s)));

		do {
			b = incr(qf, b);
		} while (!is_occupied(get_elem(qf, b)));
	}
	return s;
}

__device__ bool qf_may_contain_device(const struct quotient_filter *qf, uint64_t hash)
{
	uint64_t fq = hash_to_quotient(qf, hash);
	uint64_t fr = hash_to_remainder(qf, hash);
	uint64_t T_fq = get_elem(qf, fq);

	if (!is_occupied(T_fq)) {
		return false;
	}

	uint64_t s = find_run_index(qf, fq);
	do {
		uint64_t rem = get_remainder(get_elem(qf, s));
		if (rem == fr) {
			return true;
		} else if (rem > fr) {
			return false;
		}
		s = incr(qf, s);
	} while (is_continuation(get_elem(qf, s)));

	return false;
}

__global__ void qf_lookup_single_kernel(const struct quotient_filter *qf_struct, uint64_t hash, bool *result)
{
	*result = qf_may_contain_device(qf_struct, hash);
}

extern "C" {

extern size_t qf_table_size(uint32_t q, uint32_t r);

struct quotient_filter *qf_copy_to_gpu(struct quotient_filter *cpu_qf)
{
	cudaError_t err;

	size_t table_size = qf_table_size(cpu_qf->qf_qbits, cpu_qf->qf_rbits);

	uint64_t *d_table_array;
	err = cudaMalloc((void **)&d_table_array, table_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Malloc (Table) failed: %s\n", cudaGetErrorString(err));
		return NULL;
	}

	err = cudaMemcpy(d_table_array, cpu_qf->qf_table, table_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Memcpy (Table) failed: %s\n", cudaGetErrorString(err));
		cudaFree(d_table_array);
		return NULL;
	}

	struct quotient_filter shadow_qf = *cpu_qf;
	shadow_qf.qf_table = d_table_array;

	struct quotient_filter *d_qf_struct;
	err = cudaMalloc((void **)&d_qf_struct, sizeof(struct quotient_filter));
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Malloc (Struct) failed: %s\n", cudaGetErrorString(err));
		cudaFree(d_table_array);
		return NULL;
	}

	err = cudaMemcpy(d_qf_struct, &shadow_qf, sizeof(struct quotient_filter), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Memcpy (Struct) failed: %s\n", cudaGetErrorString(err));
		cudaFree(d_table_array);
		cudaFree(d_qf_struct);
		return NULL;
	}

	return d_qf_struct;
}

void qf_free_gpu(struct quotient_filter *d_qf)
{
	if (!d_qf)
		return;

	struct quotient_filter temp_qf;
	cudaMemcpy(&temp_qf, d_qf, sizeof(struct quotient_filter), cudaMemcpyDeviceToHost);

	cudaFree(temp_qf.qf_table);

	cudaFree(d_qf);
}

bool qf_gpu_lookup(struct quotient_filter *d_qf, uint64_t hash)
{
	bool h_result = false;

	bool *d_result;
	cudaMalloc((void **)&d_result, sizeof(bool));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	qf_lookup_single_kernel<<<1, 1>>>(d_qf, hash, d_result);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time_ms = 0;
	cudaEventElapsedTime(&time_ms, start, stop);

	printf("GPU lookup time: %f ms\n", time_ms);

	cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(d_result);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return h_result;
}
}
