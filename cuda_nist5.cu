extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void quark_blake512_cpu_init(int thr_id);
extern void quark_blake512_cpu_setBlock_80(uint64_t *pdata);
extern void quark_blake512_cpu_setBlock_80_multi(uint32_t thr_id, uint64_t *pdata);
extern void quark_blake512_cpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_80_multi(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void quark_groestl512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_skein512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

//extern void quark_jh512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
//extern void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
//extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void cuda_jh512Keccak512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_hash);


extern void quark_skein512_cpu_init(int thr_id);
extern void quark_skein512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
//extern void quark_skein512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *h_found, uint32_t target);


// Original nist5hash Funktion aus einem miner Quelltext
extern "C" void nist5hash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    
    uint8_t hash[64];

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, (void*) hash);
    
    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
//static uint32_t *h_found[MAX_GPUS];

extern "C" int scanhash_nist5(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1 << 20); // 256*256*16
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

		// Konstanten kopieren, Speicher belegen
		quark_skein512_cpu_init(thr_id);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));
//		CUDA_CALL_OR_RET_X(cudaMallocHost(&(h_found[thr_id]), 2 * sizeof(uint32_t)), 0);
		quark_blake512_cpu_init(thr_id);
		
		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	if (opt_n_gputhreads > 1)
	{
		quark_blake512_cpu_setBlock_80_multi(thr_id, (uint64_t *)endiandata);
	}
	else
	{
		quark_blake512_cpu_setBlock_80((uint64_t *)endiandata);
	}
	cuda_check_cpu_setTarget(ptarget);

	do {

		// Hash with CUDA
		if (opt_n_gputhreads > 1)
		{
			quark_blake512_cpu_hash_80_multi(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}
		else
		{
			quark_blake512_cpu_hash_80(throughput, pdata[19], d_hash[thr_id]);
		}
		quark_groestl512_cpu_hash_64(throughput, pdata[19], NULL, d_hash[thr_id]);
//		quark_jh512_cpu_hash_64(throughput, pdata[19], NULL, d_hash[thr_id]);
//		quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id]);
		cuda_jh512Keccak512_cpu_hash_64(throughput, pdata[19], d_hash[thr_id]);
//		quark_skein512_cpu_hash_64_final(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], h_found[thr_id], ptarget[7]);
		quark_skein512_cpu_hash_64(throughput, pdata[19], NULL,d_hash[thr_id]);
		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce);
			nist5hash(vhash64, endiandata);

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
