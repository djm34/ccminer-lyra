extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

// Speicher zur Generierung der Noncevektoren für die bedingten Hashes
uint32_t *d_branch1Nonces[MAX_GPUS];
uint32_t *d_branch2Nonces[MAX_GPUS];
uint32_t *d_branch3Nonces[MAX_GPUS];


extern void quark_blake512_cpu_init(int thr_id);
extern void quark_blake512_cpu_setBlock_80(uint64_t *pdata);
extern void quark_blake512_cpu_setBlock_80_multi(uint32_t thr_id, uint64_t *pdata);

extern void quark_blake512_cpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_80_multi(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void quark_blake512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_bmw512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_bmw512_cpu_hash_64_quark(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_groestl512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_skein512_cpu_init(int thr_id);
extern void quark_skein512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);

extern void quark_keccakskein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_keccak512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t target, uint32_t *h_found);
extern void quark_keccak512_cpu_init(int thr_id);


extern void quark_jh512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash);
extern void quark_jh512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t target, uint32_t *h_found);
extern void quark_jh512_cpu_init(int thr_id);

extern void quark_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void quark_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1,
											uint32_t *d_nonces2, uint32_t *nrm2);
extern void quark_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
											uint32_t *d_nonces1, uint32_t *nrm1);

extern uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash);
extern void cuda_check_quarkcoin(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, uint32_t *foundnonces);

// Original Quarkhash Funktion aus einem miner Quelltext
extern "C" void quarkhash(void *state, const void *input)
{
    sph_blake512_context ctx_blake;
    sph_bmw512_context ctx_bmw;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    
    unsigned char hash[64];

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, input, 80);
    sph_blake512_close(&ctx_blake, (void*) hash);
    
    sph_bmw512_init(&ctx_bmw);
    sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
    sph_bmw512_close(&ctx_bmw, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_groestl512_init(&ctx_groestl);
        sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
        sph_groestl512_close(&ctx_groestl, (void*) hash);
    }
    else
    {
        sph_skein512_init(&ctx_skein);
        sph_skein512 (&ctx_skein, (const void*) hash, 64);
        sph_skein512_close(&ctx_skein, (void*) hash);
    }
    
    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) hash);

    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, (const void*) hash, 64);
    sph_jh512_close(&ctx_jh, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_blake512_init(&ctx_blake);
        sph_blake512 (&ctx_blake, (const void*) hash, 64);
        sph_blake512_close(&ctx_blake, (void*) hash);
    }
    else
    {
        sph_bmw512_init(&ctx_bmw);
        sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
        sph_bmw512_close(&ctx_bmw, (void*) hash);
    }

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) hash);

    sph_skein512_init(&ctx_skein);
    sph_skein512 (&ctx_skein, (const void*) hash, 64);
    sph_skein512_close(&ctx_skein, (void*) hash);

    if (hash[0] & 0x8)
    {
        sph_keccak512_init(&ctx_keccak);
        sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
        sph_keccak512_close(&ctx_keccak, (void*) hash);
    }
    else
    {
        sph_jh512_init(&ctx_jh);
        sph_jh512 (&ctx_jh, (const void*) hash, 64);
        sph_jh512_close(&ctx_jh, (void*) hash);
    }

    memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static uint32_t endiandata[MAX_GPUS][20];
static uint32_t foundnonces[MAX_GPUS][2];
static uint32_t foundnonces2[MAX_GPUS][2];

extern "C" int scanhash_quark(int thr_id, uint32_t *pdata,
    uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	uint32_t intensity = 1 << 22;
	intensity = intensity + ((1 << 22)*9/10);
	if (device_sm[device_map[thr_id]] > 500) intensity= 1 << 24;

	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity); // 256*4096
	throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] =0x2f;

	if (!init[thr_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}

		// Konstanten kopieren, Speicher belegen
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput));

		quark_compactTest_cpu_init(thr_id, throughput);

		cudaMalloc(&d_branch1Nonces[thr_id], sizeof(uint32_t)*throughput);
		cudaMalloc(&d_branch2Nonces[thr_id], sizeof(uint32_t)*throughput);
		cudaMalloc(&d_branch3Nonces[thr_id], sizeof(uint32_t)*throughput);
		quark_blake512_cpu_init(thr_id);
		quark_keccak512_cpu_init(thr_id);
		quark_jh512_cpu_init(thr_id);
		CUDA_SAFE_CALL(cudaGetLastError());
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[thr_id][k], ((uint32_t*)pdata)[k]);
	cuda_check_cpu_setTarget(ptarget);
	if (opt_n_gputhreads > 1)
	{
		quark_blake512_cpu_setBlock_80_multi(thr_id, (uint64_t *)endiandata[thr_id]);
	}
	else
	{
		quark_blake512_cpu_setBlock_80((uint64_t *)endiandata[thr_id]);
	}

	do {

		uint32_t nrm1 = 0, nrm2 = 0, nrm3 = 0;

		if (opt_n_gputhreads > 1)
		{
			quark_blake512_cpu_hash_80_multi(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}
		else
		{
			quark_blake512_cpu_hash_80( throughput, pdata[19], d_hash[thr_id]);
		}
		quark_bmw512_cpu_hash_64_quark(throughput, pdata[19], NULL, d_hash[thr_id]);

		quark_compactTest_single_false_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], NULL,
			d_branch3Nonces[thr_id], &nrm3);

		// nur den Skein Branch weiterverfolgen
		quark_skein512_cpu_hash_64(nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// das ist der unbedingte Branch für Groestl512
		quark_groestl512_cpu_hash_64(nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// das ist der unbedingte Branch für JH512
		quark_jh512_cpu_hash_64(nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// quarkNonces in branch1 und branch2 aufsplitten gemäss if (hash[0] & 0x8)
		quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2);

		// das ist der bedingte Branch für Blake512
		quark_blake512_cpu_hash_64(nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id]);

		// das ist der bedingte Branch für Bmw512
		quark_bmw512_cpu_hash_64(nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id]);

		quark_keccakskein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id]);

		// quarkNonces in branch1 und branch2 aufsplitten gemäss if (hash[0] & 0x8)
		quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2);
		
		quark_jh512_cpu_hash_64_final(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], ptarget[7], &foundnonces[thr_id][0]);
		quark_keccak512_cpu_hash_64_final(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], ptarget[7], &foundnonces2[thr_id][0]);

		if (foundnonces[thr_id][0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[thr_id][19], foundnonces[thr_id][0]);
			quarkhash(vhash64, endiandata[thr_id]);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				// check if there was some other ones...
				if (foundnonces2[thr_id][0] != 0xffffffff)
				{
					const uint32_t Htarg = ptarget[7];
					uint32_t vhash64[8];
					be32enc(&endiandata[thr_id][19], foundnonces2[thr_id][0]);
					quarkhash(vhash64, endiandata[thr_id]);

					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = foundnonces2[thr_id][0];
						res++;
						if (opt_benchmark) applog(LOG_INFO, "GPU #%d: Found second nonce $%08X", thr_id, foundnonces2[thr_id][0]);
					}
					else
					{
						if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
							applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundnonces2[thr_id][0]);
					}
				} else	if (foundnonces[thr_id][1] != 0xffffffff)
				{
					pdata[21] = foundnonces[thr_id][1];
					res++;
					if (opt_benchmark)  applog(LOG_INFO, "GPU #%d: Found second nonce $%08X", thr_id, foundnonces[thr_id][1]);
				}				
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d: Found nonce $%08X", thr_id, foundnonces[thr_id][0]);
				pdata[19] = foundnonces[thr_id][0];
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundnonces[thr_id][0]);
			}
		}

		if (foundnonces2[thr_id][0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[thr_id][19], foundnonces2[thr_id][0]);
			quarkhash(vhash64, endiandata[thr_id]);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				// check if there was some other ones...
				if (foundnonces2[thr_id][1] != 0xffffffff)
				{
					pdata[21] = foundnonces2[thr_id][1];
					res++;
					if (opt_benchmark)  applog(LOG_INFO, "GPU #%d: Found second nonce $%08X", thr_id, foundnonces2[thr_id][1]);
				}
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d: Found nonce $%08X", thr_id, foundnonces2[thr_id][0]);
				pdata[19] = foundnonces2[thr_id][0];
				return res;
			}
			else
			{
				if (vhash64[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundnonces2[thr_id][0]);
			}
		}


		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
