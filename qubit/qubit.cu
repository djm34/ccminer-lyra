/*
 * qubit algorithm
 *
 */
extern "C" {
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}

#include "miner.h"

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *h_found[MAX_GPUS];

extern void qubit_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void qubit_luffa512_cpu_setBlock_80(void *pdata);
extern void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);



extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x11_shavite512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern int x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id,uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
//extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void x11_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t target, uint32_t *h_found);

extern void quark_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void quark_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes,
											uint32_t *d_noncesTrue, uint32_t *nrmTrue, uint32_t *d_noncesFalse, uint32_t *nrmFalse);

extern "C" void qubithash(void *state, const void *input)
{
	// luffa1-cubehash2-shavite3-simd4-echo5

	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	uint8_t hash[64];

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_qubit(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t endiandata[20];
	const uint32_t first_nonce = pdata[19];

	uint32_t intensity = 256 * 256 * 10;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);
	if (strstr(props.name, "970"))
	{
		intensity = 256 * 256 * 24;
	}
	else if (strstr(props.name, "980"))
	{
		intensity = 256 * 256 * 24;
	}
	else if (strstr(props.name, "750 Ti"))
	{
		intensity = 256 * 256 * 12;
	}
	else if (strstr(props.name, "750"))
	{
		intensity = 256 * 256 * 10;
	}
	else if (strstr(props.name, "960"))
	{
		intensity = 256 * 256 * 23;
	}
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity);

	throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0xf;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}

		qubit_luffa512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&(h_found[thr_id]), 4 * sizeof(uint32_t)), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	qubit_luffa512_cpu_setBlock_80((void*)endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {

		// Hash with CUDA
		qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]);
		x11_shavite512_cpu_hash_64(throughput, pdata[19], d_hash[thr_id]);
		x11_simd512_cpu_hash_64(thr_id,throughput, pdata[19], d_hash[thr_id]);
		x11_echo512_cpu_hash_64_final(thr_id, throughput, pdata[19], d_hash[thr_id], ptarget[7], h_found[thr_id]);
		if (h_found[thr_id][0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], h_found[thr_id][0]);
			qubithash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (h_found[thr_id][1] != 0xffffffff)
				{
					pdata[21] = h_found[thr_id][1];
					res++;
					if (opt_benchmark)
						applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, h_found[thr_id][1], vhash64[7], Htarg);
				}
				pdata[19] = h_found[thr_id][0];
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, h_found[thr_id][0], vhash64[7], Htarg);
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg)
				{
					applog(LOG_INFO, "GPU #%d: result for %08x does not validate on CPU!", thr_id, h_found[thr_id][0]);
				}
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
