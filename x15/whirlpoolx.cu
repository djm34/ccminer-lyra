/*
 * whirlpool routine (djm)
 * whirlpoolx routine (provos alexis)
 */
extern "C"
{
#include "sph/sph_whirlpool.h"
#include "miner.h"
}

#include "cuda_helper.h"

extern void whirlpoolx_cpu_init(int thr_id, uint32_t threads);
extern void whirlpoolx_setBlock_80(void *pdata, const void *ptarget);
extern void cpu_whirlpoolx(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *foundNonce);
extern void whirlpoolx_precompute(int thr_id);

// CPU Hash function
extern "C" void whirlxHash(void *state, const void *input)
{

	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[64];
	unsigned char hash_xored[32];

	memset(hash, 0, sizeof(hash));

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, input, 80);
	sph_whirlpool_close(&ctx_whirlpool, hash);

    
	for (uint32_t i = 0; i < 32; i++){
	        hash_xored[i] = hash[i] ^ hash[i + 16];
	}
	memcpy(state, hash_xored, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_whirlpoolx(int thr_id, uint32_t *pdata, uint32_t *ptarget, uint32_t max_nonce, uint32_t *hashes_done)
{
	uint32_t foundNonce[MAX_GPUS][4];
	const uint32_t first_nonce = pdata[19];
	uint32_t endiandata[20];
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, (1 << 25));
	throughput = min(throughput, max_nonce - first_nonce);
	applog(LOG_INFO, "debug #%d: throughput %08x", thr_id, pdata[19]);
	if (opt_benchmark)
		ptarget[7] = 0x5;

	if (!init[thr_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		whirlpoolx_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
	{
		be32enc(&endiandata[k], pdata[k]);
	}

	whirlpoolx_setBlock_80((void*)endiandata, &ptarget[6]);
	whirlpoolx_precompute(thr_id);
	do {
		cpu_whirlpoolx(thr_id, throughput, pdata[19], foundNonce[thr_id]);
//		CUDA_SAFE_CALL(cudaGetLastError());
		if (foundNonce[thr_id][0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			/* check now with the CPU to confirm */
			be32enc(&endiandata[19], foundNonce[thr_id][0]);
			whirlxHash(vhash64, endiandata);
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
			/*		if (foundNonce[thr_id][1] != UINT32_MAX)
				{
					be32enc(&endiandata[19], foundNonce[thr_id][1]);
					whirlxHash(vhash64, endiandata);
					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundNonce[thr_id][1];
						res++;
						if (opt_benchmark) applog(LOG_INFO, "GPU #%d: found nonce %08x", thr_id, foundNonce[thr_id][1]);
					}
					else
					{
						if (vhash64[7] != Htarg)
							applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce[thr_id][1]);
					}
				}
				*/

				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d: found nonce %08x", thr_id, foundNonce[thr_id][0], vhash64[7]);
				pdata[19] = foundNonce[thr_id][0];
				return res;
			}
			else
			{
				if(vhash64[7] != Htarg)
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", thr_id, foundNonce[thr_id][0]);
			}
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
