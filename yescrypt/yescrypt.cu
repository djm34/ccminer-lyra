
extern "C"
{
#include "sph/yescrypt.h"
}

#include "cuda_helper.h"
#include "miner.h"


static uint32_t *d_hash[MAX_GPUS] ;
static uint32_t *d_hash2[MAX_GPUS];
static uint32_t *d_hash3[MAX_GPUS];
static uint32_t *d_hash4[MAX_GPUS];


extern void yescrypt_setBlockTarget(uint32_t * data, const void *ptarget);
extern void yescrypt_cpu_init(int thr_id, int threads, uint32_t* hash, uint32_t* hash2, uint32_t* hash3, uint32_t* hash4);
extern uint32_t yescrypt_cpu_hash_k4(int thr_id, int threads, uint32_t startNounce, int order);
  

extern "C" int scanhash_yescrypt(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
//	if (pdata[0] == 0) {return 0;} // don't start unless it is really up

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	const uint32_t Htarg = ptarget[7];

//	const int throughput = gpus_intensity[thr_id] ? 256 * 64 * gpus_intensity[thr_id] : 256 * 64 * 3.5;
	int coef = 2;
	if (device_sm[device_map[thr_id]] == 500) coef = 1;
	if (device_sm[device_map[thr_id]] == 350) coef = 2;

	const int throughput = 256*coef;

	static bool init[MAX_GPUS] = { 0 };
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]); 
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		 
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id],  2048 * 128 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash2[thr_id], 8 * sizeof(uint32_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash3[thr_id], 32*64 * sizeof(uint32_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash4[thr_id], 32*8 * sizeof(uint32_t) * throughput));


 
		yescrypt_cpu_init(thr_id, throughput, d_hash[thr_id], d_hash2[thr_id], d_hash3[thr_id], d_hash4[thr_id]);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	yescrypt_setBlockTarget(pdata,ptarget);

	do {
		int order = 0;
		uint32_t foundNonce = yescrypt_cpu_hash_k4(thr_id, throughput, pdata[19], order++);
 //       uint32_t foundNonce = 0;
//		foundNonce = 10 + pdata[19];
		if  (foundNonce != 0xffffffff)
		{
			uint32_t vhash64[8];
			 
				 be32enc(&endiandata[19], foundNonce);
//	for (int i = 0; i<10; i++) { printf("i=%d endiandata %08x %08x\n",i, endiandata[2 * i], endiandata[2 * i + 1]); }

//			yescrypt_hash((unsigned char*) endiandata, (unsigned char*)vhash64);

//			if ( vhash64[7] <= ptarget[7]) { // && fulltest(vhash64, ptarget)) {
				pdata[19] = foundNonce;
				*hashes_done = foundNonce - first_nonce + 1;
				return 1;
/*
			} else {
			*hashes_done = foundNonce - first_nonce + 1; // keeps hashrate calculation happy
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
			}
*/
		}

		pdata[19] += throughput;
    } while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
