#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>

#include "uint256.h"
#include "sph/sph_groestl.h"

#include "miner.h"
#include <cuda_runtime.h>

static bool init[MAX_GPUS] = { 0 };
static uint32_t *h_found[MAX_GPUS];

void myriadgroestl_cpu_init(int thr_id, uint32_t threads);
void myriadgroestl_cpu_setBlock(int thr_id, void *data, void *pTargetIn);
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *nounce);

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

extern "C" void myriadhash(void *state, const void *input)
{
	uint32_t hashA[16], hashB[16];
	sph_groestl512_context ctx_groestl;
	SHA256_CTX sha256;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, input, 80);
	sph_groestl512_close(&ctx_groestl, hashA);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256,(unsigned char *)hashA, 64);
	SHA256_Final((unsigned char *)hashB, &sha256);

	memcpy(state, hashB, 32);
}

extern "C" int scanhash_myriad(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t start_nonce = pdata[19]++;
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1 << 20);
	throughput = min(throughput, max_nonce - start_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0ff;

	// init
	if(!init[thr_id])
	{
#if BIG_DEBUG
#else
		myriadgroestl_cpu_init(thr_id, throughput);
#endif
		cudaMallocHost(&(h_found[thr_id]), 4 * sizeof(uint32_t));
		init[thr_id] = true;
	}

	uint32_t endiandata[32];
	for (int kk=0; kk < 32; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
	myriadgroestl_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		const uint32_t Htarg = ptarget[7];

		myriadgroestl_cpu_hash(thr_id, throughput, pdata[19], h_found[thr_id]);

		if (h_found[thr_id][0] < 0xffffffff)
		{
			uint32_t tmpHash[8];
			endiandata[19] = SWAP32(h_found[thr_id][0]);
			myriadhash(tmpHash, endiandata);
			if (tmpHash[7] <= Htarg && fulltest(tmpHash, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - start_nonce + throughput;
				if (h_found[thr_id][1] != 0xffffffff)
				{
					if (opt_benchmark) applog(LOG_INFO, "found second nounce %08x", thr_id, h_found[thr_id][1]);
					pdata[21] = h_found[thr_id][1];
					res++;
				}
				pdata[19] = h_found[thr_id][0];
				if (opt_benchmark)
					applog(LOG_INFO, "found nounce %08x", thr_id, h_found[thr_id][0]);
				return res;
			}
			else
			{
				if (tmpHash[7] != Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", thr_id, h_found[thr_id][0]);
			}
		}

		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - start_nonce + 1;
	return 0;
}

