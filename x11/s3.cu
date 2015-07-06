/**
 * S3 Hash (Also called 3S - Used by 1Coin)
 */

extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include <stdint.h>

static uint32_t *d_hash[MAX_GPUS];

extern void x11_shavite512_cpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void x11_shavite512_setBlock_80(void *pdata);

extern int  x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void quark_skein512_cpu_init(int thr_id);
extern void quark_skein512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
//extern void quark_skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *h_found, uint32_t target);
extern void quark_skein512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

/* CPU HASH */
extern "C" void s3hash(void *output, const void *input)
{
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_skein512_context ctx_skein;

	unsigned char hash[64];

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, input, 80);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

/* Main S3 entry point */
extern "C" int scanhash_s3(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	unsigned int intensity = 20; // 256*256*8*2;
#ifdef WIN32
	// reduce by one the intensity on windows
	intensity--;
#endif
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1 << intensity);
	throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000000fu;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}

		x11_simd512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	x11_shavite512_setBlock_80((void*)endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {

		x11_shavite512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
		x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]);
		quark_skein512_cpu_hash_64(throughput, pdata[19], NULL, d_hash[thr_id]);
		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);

		if (foundNonce != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			s3hash(vhash64, endiandata);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				/*
				if (h_found[thr_id][1] != 0xffffffff)
				{
					pdata[21] = h_found[thr_id][1];
					res++;
					if (opt_benchmark)
						applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, h_found[thr_id][1]);
				}
				*/
				pdata[19] = foundNonce;
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", thr_id, foundNonce);
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg)
				{
					applog(LOG_INFO, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce);
				}
			}
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
