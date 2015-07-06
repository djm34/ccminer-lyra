/**
 * Blake-256 Cuda Kernel (Tested on SM 5.0)
 *
 * Tanguy Pruvot - Nov. 2014
 */
extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda_helper.h"

#include <memory.h>

static __device__ uint64_t cuda_swab32ll(uint64_t x) {
	return MAKE_ULONGLONG(cuda_swab32(_LOWORD(x)), cuda_swab32(_HIWORD(x)));
}

__constant__ static uint32_t  c_data[20];

__constant__ static uint8_t sigma[16][16];
static uint8_t  c_sigma[16][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

static const uint32_t  c_IV256[8] = {
	0x6A09E667, 0xBB67AE85,
	0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C,
	0x1F83D9AB, 0x5BE0CD19
};

__device__ __constant__ static uint32_t cpu_h[8];

__device__ __constant__ static  uint32_t  u256[16];
static const uint32_t  c_u256[16] = {
	0x243F6A88, 0x85A308D3,
	0x13198A2E, 0x03707344,
	0xA4093822, 0x299F31D0,
	0x082EFA98, 0xEC4E6C89,
	0x452821E6, 0x38D01377,
	0xBE5466CF, 0x34E90C6C,
	0xC0AC29B7, 0xC97C50DD,
	0x3F84D5B5, 0xB5470917
};

#define GS2(a,b,c,d,x) { \
	const uint8_t idx1 = sigma[r][x]; \
	const uint8_t idx2 = sigma[r][x+1]; \
	v[a] += (m[idx1] ^ u256[idx2]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ u256[idx1]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
}

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define hostGS(a,b,c,d,x) { \
	const uint8_t idx1 = c_sigma[r][x]; \
	const uint8_t idx2 = c_sigma[r][x+1]; \
	v[a] += (m[idx1] ^ c_u256[idx2]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ c_u256[idx1]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
	}

#define GSPREC(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ u256[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a],0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
		}

__host__ __forceinline__
static void blake256_compress1st(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t m[16];
	uint32_t v[16];
	
	for (int i = 0; i < 16; i++) {
		m[i] = block[i];
	}

	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] = c_u256[0];
	v[9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	for (int r = 0; r < 14; r++) {
		/* column step */
		hostGS(0, 4, 0x8, 0xC, 0x0);
		hostGS(1, 5, 0x9, 0xD, 0x2);
		hostGS(2, 6, 0xA, 0xE, 0x4);
		hostGS(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		hostGS(0, 5, 0xA, 0xF, 0x8);
		hostGS(1, 6, 0xB, 0xC, 0xA);
		hostGS(2, 7, 0x8, 0xD, 0xC);
		hostGS(3, 4, 0x9, 0xE, 0xE);
	}

	h[0] ^= v[0] ^ v[8];
	h[1] ^= v[1] ^ v[9];
	h[2] ^= v[2] ^ v[10];
	h[3] ^= v[3] ^ v[11];
	h[4] ^= v[4] ^ v[12];
	h[5] ^= v[5] ^ v[13];
	h[6] ^= v[6] ^ v[14];
	h[7] ^= v[7] ^ v[15];
}

__device__ __forceinline__
static void blake256_compress2nd(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t v[16];

	const uint32_t c_Padding[12] = {
		0x80000000, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 0, 640
	};

	uint32_t m[16]=
	{
		block[0], block[1], block[2], block[3],
		c_Padding[0], c_Padding[1], c_Padding[2], c_Padding[3],
		c_Padding[4], c_Padding[5], c_Padding[6], c_Padding[7],
		c_Padding[8], c_Padding[9], c_Padding[10], c_Padding[11]
	};

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] =  u256[0];
	v[9] =  u256[1];
	v[10] = u256[2];
	v[11] = u256[3];
	v[12] = u256[4] ^ T0;
	v[13] = u256[5] ^ T0;
	v[14] = u256[6];
	v[15] = u256[7];

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC, 0, 1);
	GSPREC(1, 5, 0x9, 0xD, 2, 3);
	GSPREC(2, 6, 0xA, 0xE, 4, 5);
	GSPREC(3, 7, 0xB, 0xF, 6, 7);
	GSPREC(0, 5, 0xA, 0xF, 8, 9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);
	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);
	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);

	//	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	GSPREC(0, 4, 0x8, 0xC, 9, 0);
	GSPREC(1, 5, 0x9, 0xD, 5, 7);
	GSPREC(2, 6, 0xA, 0xE, 2, 4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6, 8);
	GSPREC(3, 4, 0x9, 0xE, 3, 13);
	//	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15, 14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);

	//	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14, 13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0, 7);
	GSPREC(1, 6, 0xB, 0xC, 6, 3);
	GSPREC(2, 7, 0x8, 0xD, 9, 2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);

	//	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	GSPREC(0, 4, 0x8, 0xC, 13, 11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3, 9);
	GSPREC(0, 5, 0xA, 0xF, 5, 0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8, 6);
	GSPREC(3, 4, 0x9, 0xE, 2, 10);
	//	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	GSPREC(0, 4, 0x8, 0xC, 6, 15);
	GSPREC(1, 5, 0x9, 0xD, 14, 9);
	GSPREC(2, 6, 0xA, 0xE, 11, 3);
	GSPREC(3, 7, 0xB, 0xF, 0, 8);
	GSPREC(0, 5, 0xA, 0xF, 12, 2);
	GSPREC(1, 6, 0xB, 0xC, 13, 7);
	GSPREC(2, 7, 0x8, 0xD, 1, 4);
	GSPREC(3, 4, 0x9, 0xE, 10, 5);
	//	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	GSPREC(0, 4, 0x8, 0xC, 10, 2);
	GSPREC(1, 5, 0x9, 0xD, 8, 4);
	GSPREC(2, 6, 0xA, 0xE, 7, 6);
	GSPREC(3, 7, 0xB, 0xF, 1, 5);
	GSPREC(0, 5, 0xA, 0xF, 15, 11);
	GSPREC(1, 6, 0xB, 0xC, 9, 14);
	GSPREC(2, 7, 0x8, 0xD, 3, 12);
	GSPREC(3, 4, 0x9, 0xE, 13, 0);
	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	GSPREC(0, 4, 0x8, 0xC, 0, 1);
	GSPREC(1, 5, 0x9, 0xD, 2, 3);
	GSPREC(2, 6, 0xA, 0xE, 4, 5);
	GSPREC(3, 7, 0xB, 0xF, 6, 7);
	GSPREC(0, 5, 0xA, 0xF, 8, 9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);

	//	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4, 8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0, 2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5, 3);

	//	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5, 2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3, 6);
	GSPREC(2, 7, 0x8, 0xD, 7, 1);
	GSPREC(3, 4, 0x9, 0xE, 9, 4);
	//	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	GSPREC(0, 4, 0x8, 0xC, 7, 9);
	GSPREC(1, 5, 0x9, 0xD, 3, 1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2, 6);
	GSPREC(1, 6, 0xB, 0xC, 5, 10);
	GSPREC(2, 7, 0x8, 0xD, 4, 0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);


	h[0] ^= v[0] ^ v[8];
	h[1] ^= v[1] ^ v[9];
	h[2] ^= v[2] ^ v[10];
	h[3] ^= v[3] ^ v[11];
	h[4] ^= v[4] ^ v[12];
	h[5] ^= v[5] ^ v[13];
	h[6] ^= v[6] ^ v[14];
	h[7] ^= v[7] ^ v[15];
}

__global__ __launch_bounds__(256,4)
void blake256_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t * Hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8];
		uint32_t input[4];

		#pragma unroll 8
		for (int i = 0; i<8; i++) { h[i] = cpu_h[i];}

		#pragma unroll 3
		for (int i = 0; i < 3; ++i) input[i] = c_data[16 + i];

		input[3] = nonce;
		blake256_compress2nd(h, input, 640);

        #pragma unroll
		for (int i = 0; i<4; i++) {
			Hash[i*threads + thread] = cuda_swab32ll(MAKE_ULONGLONG(h[2 * i], h[2*i+1]));
		}
	}
}

__host__
void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	blake256_gpu_hash_80 <<<grid, block>>> (threads, startNonce, Hash);
}

__host__
void blake256_cpu_setBlock_80(uint32_t *pdata)
{
	uint32_t h[8];
	uint32_t data[20];
	memcpy(data, pdata, 80);
	for (int i = 0; i<8; i++) {
		h[i] = c_IV256[i];
	}
	blake256_compress1st(h, pdata, 512);

	cudaMemcpyToSymbol(cpu_h, h, sizeof(h), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, data, sizeof(data), 0, cudaMemcpyHostToDevice);
}

__host__
void blake256_cpu_init(int thr_id, uint32_t threads)
{
	cudaMemcpyToSymbol(u256, c_u256, sizeof(c_u256), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(sigma, c_sigma, sizeof(c_sigma), 0, cudaMemcpyHostToDevice);
}
