#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#define ROTR(x,n) ROTR64(x,n)

#define USE_SHUFFLE 0

// die Message it Padding zur Berechnung auf der GPU
static uint2* c_PaddedMessage80[MAX_GPUS]; // padded message (80 bytes + padding)
__constant__ uint2 c_PaddedM[16];

// ---------------------------- BEGIN CUDA quark_blake512 functions ------------------------------------

#define Gprecalc(a,b,c,d,idx1,idx2) { \
	v[a] += (block[idx2] ^ u512[idx1]) + v[b]; \
	v[d] = SWAPDWORDS2( v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 25); \
	v[a] += (block[idx1] ^ u512[idx2]) + v[b]; \
	v[d] = ROR16(v[d] ^ v[a]); \
	v[c] += v[d]; \
	v[b] = ROR2(v[b] ^ v[c], 11); \
	}

__global__ 
#if __CUDA_ARCH__ > 500
	__launch_bounds__(256, 1)
#else
	__launch_bounds__(256, 2)
#endif
void quark_blake512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ g_nonceVector, uint64_t *const __restrict__ g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

#if USE_SHUFFLE
	const int warpID = threadIdx.x & 0x0F; // 16 warps
	const int warpBlockID = (thread + 15)>>4; // aufrunden auf volle Warp-Blöcke
	const int maxHashPosition = thread<<3;
#endif

#if USE_SHUFFLE
	if (warpBlockID < ( (threads+15)>>4 ))
#else
	if (thread < threads)
#endif
	{
		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		const int hashPosition = nounce - startNounce;

		uint64_t *inpHash = &g_hash[hashPosition*8];
		uint2 block[16];
		block[0] = vectorizeswap(inpHash[0]);
		block[1] = vectorizeswap(inpHash[1]);
		block[2] = vectorizeswap(inpHash[2]);
		block[3] = vectorizeswap(inpHash[3]);
		block[4] = vectorizeswap(inpHash[4]);
		block[5] = vectorizeswap(inpHash[5]);
		block[6] = vectorizeswap(inpHash[6]);
		block[7] = vectorizeswap(inpHash[7]);
		block[8] = vectorizehigh(0x80000000);
		block[9] = vectorizelow(0x0);
		block[10] = vectorizelow(0x0);
		block[11] = vectorizelow(0x0);
		block[12] = vectorizelow(0x0);
		block[13] = vectorizelow(0x1);
		block[14] = vectorizelow(0x0);
		block[15] = vectorizelow(0x200);

		const uint2 h[8] =
		{
				{ 0xf3bcc908UL, 0x6a09e667UL },
				{ 0x84caa73bUL, 0xbb67ae85UL },
				{ 0xfe94f82bUL, 0x3c6ef372UL },
				{ 0x5f1d36f1UL, 0xa54ff53aUL },
				{ 0xade682d1UL, 0x510e527fUL },
				{ 0x2b3e6c1fUL, 0x9b05688cUL },
				{ 0xfb41bd6bUL, 0x1f83d9abUL },
				{ 0x137e2179UL, 0x5be0cd19UL }
		};
		const uint2 u512[16] =
		{
			{ 0x85a308d3UL, 0x243f6a88 }, { 0x03707344UL, 0x13198a2e },
			{ 0x299f31d0UL, 0xa4093822 }, { 0xec4e6c89UL, 0x082efa98 },
			{ 0x38d01377UL, 0x452821e6 }, { 0x34e90c6cUL, 0xbe5466cf },
			{ 0xc97c50ddUL, 0xc0ac29b7 }, { 0xb5470917UL, 0x3f84d5b5 },
			{ 0x8979fb1bUL, 0x9216d5d9 }, { 0x98dfb5acUL, 0xd1310ba6 },
			{ 0xd01adfb7UL, 0x2ffd72db }, { 0x6a267e96UL, 0xb8e1afed },
			{ 0xf12c7f99UL, 0xba7c9045 }, { 0xb3916cf7UL, 0x24a19947 },
			{ 0x858efc16UL, 0x0801f2e2 }, { 0x71574e69UL, 0x636920d8 }
		};

		uint2 v[16] =
		{
			h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
			u512[0], u512[1], u512[2], u512[3], u512[4] ^ 512, u512[5] ^ 512, u512[6], u512[7]
		};

		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
		Gprecalc(1, 5, 9, 13, 0x3, 0x2)
		Gprecalc(2, 6, 10, 14, 0x5, 0x4)
		Gprecalc(3, 7, 11, 15, 0x7, 0x6)
		Gprecalc(0, 5, 10, 15, 0x9, 0x8)
		Gprecalc(1, 6, 11, 12, 0xb, 0xa)
		Gprecalc(2, 7, 8, 13, 0xd, 0xc)
		Gprecalc(3, 4, 9, 14, 0xf, 0xe)

		Gprecalc(0, 4, 8, 12, 0xa, 0xe)
		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		Gprecalc(0, 5, 10, 15, 0xc, 0x1)
		Gprecalc(1, 6, 11, 12, 0x2, 0x0)
		Gprecalc(2, 7, 8, 13, 0x7, 0xb)
		Gprecalc(3, 4, 9, 14, 0x3, 0x5)

		Gprecalc(0, 4, 8, 12, 0x8, 0xb)
		Gprecalc(1, 5, 9, 13, 0x0, 0xc)
		Gprecalc(2, 6, 10, 14, 0x2, 0x5)
		Gprecalc(3, 7, 11, 15, 0xd, 0xf)
		Gprecalc(0, 5, 10, 15, 0xe, 0xa)
		Gprecalc(1, 6, 11, 12, 0x6, 0x3)
		Gprecalc(2, 7, 8, 13, 0x1, 0x7)
		Gprecalc(3, 4, 9, 14, 0x4, 0x9)
		
		Gprecalc(0, 4, 8, 12, 0x9, 0x7)
		Gprecalc(1, 5, 9, 13, 0x1, 0x3)
		Gprecalc(2, 6, 10, 14, 0xc, 0xd)
		Gprecalc(3, 7, 11, 15, 0xe, 0xb)
		Gprecalc(0, 5, 10, 15, 0x6, 0x2)
		Gprecalc(1, 6, 11, 12, 0xa, 0x5)
		Gprecalc(2, 7, 8, 13, 0x0, 0x4)
		Gprecalc(3, 4, 9, 14, 0x8, 0xf)
		
		Gprecalc(0, 4, 8, 12, 0x0, 0x9)
		Gprecalc(1, 5, 9, 13, 0x7, 0x5)
		Gprecalc(2, 6, 10, 14, 0x4, 0x2)
		Gprecalc(3, 7, 11, 15, 0xf, 0xa)
		Gprecalc(0, 5, 10, 15, 0x1, 0xe)
		Gprecalc(1, 6, 11, 12, 0xc, 0xb)
		Gprecalc(2, 7, 8, 13, 0x8, 0x6)
		Gprecalc(3, 4, 9, 14, 0xd, 0x3)
		
		Gprecalc(0, 4, 8, 12, 0xc, 0x2)
		Gprecalc(1, 5, 9, 13, 0xa, 0x6)
		Gprecalc(2, 6, 10, 14, 0xb, 0x0)
		Gprecalc(3, 7, 11, 15, 0x3, 0x8)
		Gprecalc(0, 5, 10, 15, 0xd, 0x4)
		Gprecalc(1, 6, 11, 12, 0x5, 0x7)
		Gprecalc(2, 7, 8, 13, 0xe, 0xf)
		Gprecalc(3, 4, 9, 14, 0x9, 0x1)
		
		Gprecalc(0, 4, 8, 12, 0x5, 0xc)
		Gprecalc(1, 5, 9, 13, 0xf, 0x1)
		Gprecalc(2, 6, 10, 14, 0xd, 0xe)
		Gprecalc(3, 7, 11, 15, 0xa, 0x4)
		Gprecalc(0, 5, 10, 15, 0x7, 0x0)
		Gprecalc(1, 6, 11, 12, 0x3, 0x6)
		Gprecalc(2, 7, 8, 13, 0x2, 0x9)
		Gprecalc(3, 4, 9, 14, 0xb, 0x8)
		
		Gprecalc(0, 4, 8, 12, 0xb, 0xd)
		Gprecalc(1, 5, 9, 13, 0xe, 0x7)
		Gprecalc(2, 6, 10, 14, 0x1, 0xc)
		Gprecalc(3, 7, 11, 15, 0x9, 0x3)
		Gprecalc(0, 5, 10, 15, 0x0, 0x5)
		Gprecalc(1, 6, 11, 12, 0x4, 0xf)
		Gprecalc(2, 7, 8, 13, 0x6, 0x8)
		Gprecalc(3, 4, 9, 14, 0xa, 0x2)
		
		Gprecalc(0, 4, 8, 12, 0xf, 0x6)
		Gprecalc(1, 5, 9, 13, 0x9, 0xe)
		Gprecalc(2, 6, 10, 14, 0x3, 0xb)
		Gprecalc(3, 7, 11, 15, 0x8, 0x0)
		Gprecalc(0, 5, 10, 15, 0x2, 0xc)
		Gprecalc(1, 6, 11, 12, 0x7, 0xd)
		Gprecalc(2, 7, 8, 13, 0x4, 0x1)
		Gprecalc(3, 4, 9, 14, 0x5, 0xa)
		
		Gprecalc(0, 4, 8, 12, 0x2, 0xa)
		Gprecalc(1, 5, 9, 13, 0x4, 0x8)
		Gprecalc(2, 6, 10, 14, 0x6, 0x7)
		Gprecalc(3, 7, 11, 15, 0x5, 0x1)
		Gprecalc(0, 5, 10, 15, 0xb, 0xf)
		Gprecalc(1, 6, 11, 12, 0xe, 0x9)
		Gprecalc(2, 7, 8, 13, 0xc, 0x3)
		Gprecalc(3, 4, 9, 14, 0x0, 0xd)

		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
		Gprecalc(1, 5, 9, 13, 0x3, 0x2)
		Gprecalc(2, 6, 10, 14, 0x5, 0x4)
		Gprecalc(3, 7, 11, 15, 0x7, 0x6)
		Gprecalc(0, 5, 10, 15, 0x9, 0x8)
		Gprecalc(1, 6, 11, 12, 0xb, 0xa)
		Gprecalc(2, 7, 8, 13, 0xd, 0xc)
		Gprecalc(3, 4, 9, 14, 0xf, 0xe)

		Gprecalc(0, 4, 8, 12, 0xa, 0xe)
		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		Gprecalc(0, 5, 10, 15, 0xc, 0x1)
		Gprecalc(1, 6, 11, 12, 0x2, 0x0)
		Gprecalc(2, 7, 8, 13, 0x7, 0xb)
		Gprecalc(3, 4, 9, 14, 0x3, 0x5)

		Gprecalc(0, 4, 8, 12, 0x8, 0xb)
		Gprecalc(1, 5, 9, 13, 0x0, 0xc)
		Gprecalc(2, 6, 10, 14, 0x2, 0x5)
		Gprecalc(3, 7, 11, 15, 0xd, 0xf)
		Gprecalc(0, 5, 10, 15, 0xe, 0xa)
		Gprecalc(1, 6, 11, 12, 0x6, 0x3)
		Gprecalc(2, 7, 8, 13, 0x1, 0x7)
		Gprecalc(3, 4, 9, 14, 0x4, 0x9)

		Gprecalc(0, 4, 8, 12, 0x9, 0x7)
		Gprecalc(1, 5, 9, 13, 0x1, 0x3)
		Gprecalc(2, 6, 10, 14, 0xc, 0xd)
		Gprecalc(3, 7, 11, 15, 0xe, 0xb)
		Gprecalc(0, 5, 10, 15, 0x6, 0x2)
		Gprecalc(1, 6, 11, 12, 0xa, 0x5)
		Gprecalc(2, 7, 8, 13, 0x0, 0x4)
		Gprecalc(3, 4, 9, 14, 0x8, 0xf)

		Gprecalc(0, 4, 8, 12, 0x0, 0x9)
		Gprecalc(1, 5, 9, 13, 0x7, 0x5)
		Gprecalc(2, 6, 10, 14, 0x4, 0x2)
		Gprecalc(3, 7, 11, 15, 0xf, 0xa)
		Gprecalc(0, 5, 10, 15, 0x1, 0xe)
		Gprecalc(1, 6, 11, 12, 0xc, 0xb)
		Gprecalc(2, 7, 8, 13, 0x8, 0x6)
		Gprecalc(3, 4, 9, 14, 0xd, 0x3)

		Gprecalc(0, 4, 8, 12, 0xc, 0x2)
		Gprecalc(1, 5, 9, 13, 0xa, 0x6)
		Gprecalc(2, 6, 10, 14, 0xb, 0x0)
		Gprecalc(3, 7, 11, 15, 0x3, 0x8)
		Gprecalc(0, 5, 10, 15, 0xd, 0x4)
		Gprecalc(1, 6, 11, 12, 0x5, 0x7)
		Gprecalc(2, 7, 8, 13, 0xe, 0xf)
		Gprecalc(3, 4, 9, 14, 0x9, 0x1)

		uint64_t *outHash = &g_hash[8 * hashPosition];

		outHash[0] = devectorizeswap(h[0] ^ v[0] ^ v[8]);
		outHash[1] = devectorizeswap(h[1] ^ v[1] ^ v[9]);
		outHash[2] = devectorizeswap(h[2] ^ v[2] ^ v[10]);
		outHash[3] = devectorizeswap(h[3] ^ v[3] ^ v[11]);
		outHash[4] = devectorizeswap(h[4] ^ v[4] ^ v[12]);
		outHash[5] = devectorizeswap(h[5] ^ v[5] ^ v[13]);
		outHash[6] = devectorizeswap(h[6] ^ v[6] ^ v[14]);
		outHash[7] = devectorizeswap(h[7] ^ v[7] ^ v[15]);
	}
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(256, 4)
#else
__launch_bounds__(32, 32)
#endif
void quark_blake512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		uint2 block[16];

		block[0] = c_PaddedM[0];
		block[1] = c_PaddedM[1];
		block[2] = c_PaddedM[2];
		block[3] = c_PaddedM[3];
		block[4] = c_PaddedM[4];
		block[5] = c_PaddedM[5];
		block[6] = c_PaddedM[6];
		block[7] = c_PaddedM[7];
		block[8] = c_PaddedM[8];
		block[9] = c_PaddedM[9];
		block[10] = vectorizehigh(0x80000000);
		block[11] = vectorizelow(0);
		block[12] = vectorizelow(0);
		block[13] = vectorizelow(0x1);
		block[14] = vectorizelow(0);
		block[15] = vectorizelow(0x280);
		block[9].x = nounce;
		const uint2 u512[16] =
		{
			{ 0x85a308d3UL, 0x243f6a88 }, { 0x03707344UL, 0x13198a2e },
			{ 0x299f31d0UL, 0xa4093822 }, { 0xec4e6c89UL, 0x082efa98 },
			{ 0x38d01377UL, 0x452821e6 }, { 0x34e90c6cUL, 0xbe5466cf },
			{ 0xc97c50ddUL, 0xc0ac29b7 }, { 0xb5470917UL, 0x3f84d5b5 },
			{ 0x8979fb1bUL, 0x9216d5d9 }, { 0x98dfb5acUL, 0xd1310ba6 },
			{ 0xd01adfb7UL, 0x2ffd72db }, { 0x6a267e96UL, 0xb8e1afed },
			{ 0xf12c7f99UL, 0xba7c9045 }, { 0xb3916cf7UL, 0x24a19947 },
			{ 0x858efc16UL, 0x0801f2e2 }, { 0x71574e69UL, 0x636920d8 }
		};

		const uint2 h[8] = {
				{ 0xf3bcc908UL, 0x6a09e667UL },
				{ 0x84caa73bUL, 0xbb67ae85UL },
				{ 0xfe94f82bUL, 0x3c6ef372UL },
				{ 0x5f1d36f1UL, 0xa54ff53aUL },
				{ 0xade682d1UL, 0x510e527fUL },
				{ 0x2b3e6c1fUL, 0x9b05688cUL },
				{ 0xfb41bd6bUL, 0x1f83d9abUL },
				{ 0x137e2179UL, 0x5be0cd19UL }
		};

		uint2 v[16] =
		{
			h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
			u512[0], u512[1], u512[2], u512[3], u512[4] ^ 640, u512[5] ^ 640, u512[6], u512[7]
		};

		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
			Gprecalc(1, 5, 9, 13, 0x3, 0x2)
			Gprecalc(2, 6, 10, 14, 0x5, 0x4)
			Gprecalc(3, 7, 11, 15, 0x7, 0x6)
			Gprecalc(0, 5, 10, 15, 0x9, 0x8)
			Gprecalc(1, 6, 11, 12, 0xb, 0xa)
			Gprecalc(2, 7, 8, 13, 0xd, 0xc)
			Gprecalc(3, 4, 9, 14, 0xf, 0xe)

			Gprecalc(0, 4, 8, 12, 0xa, 0xe)
			Gprecalc(1, 5, 9, 13, 0x8, 0x4)
			Gprecalc(2, 6, 10, 14, 0xf, 0x9)
			Gprecalc(3, 7, 11, 15, 0x6, 0xd)
			Gprecalc(0, 5, 10, 15, 0xc, 0x1)
			Gprecalc(1, 6, 11, 12, 0x2, 0x0)
			Gprecalc(2, 7, 8, 13, 0x7, 0xb)
			Gprecalc(3, 4, 9, 14, 0x3, 0x5)

			Gprecalc(0, 4, 8, 12, 0x8, 0xb)
			Gprecalc(1, 5, 9, 13, 0x0, 0xc)
			Gprecalc(2, 6, 10, 14, 0x2, 0x5)
			Gprecalc(3, 7, 11, 15, 0xd, 0xf)
			Gprecalc(0, 5, 10, 15, 0xe, 0xa)
			Gprecalc(1, 6, 11, 12, 0x6, 0x3)
			Gprecalc(2, 7, 8, 13, 0x1, 0x7)
			Gprecalc(3, 4, 9, 14, 0x4, 0x9)

			Gprecalc(0, 4, 8, 12, 0x9, 0x7)
			Gprecalc(1, 5, 9, 13, 0x1, 0x3)
			Gprecalc(2, 6, 10, 14, 0xc, 0xd)
			Gprecalc(3, 7, 11, 15, 0xe, 0xb)
			Gprecalc(0, 5, 10, 15, 0x6, 0x2)
			Gprecalc(1, 6, 11, 12, 0xa, 0x5)
			Gprecalc(2, 7, 8, 13, 0x0, 0x4)
			Gprecalc(3, 4, 9, 14, 0x8, 0xf)

			Gprecalc(0, 4, 8, 12, 0x0, 0x9)
			Gprecalc(1, 5, 9, 13, 0x7, 0x5)
			Gprecalc(2, 6, 10, 14, 0x4, 0x2)
			Gprecalc(3, 7, 11, 15, 0xf, 0xa)
			Gprecalc(0, 5, 10, 15, 0x1, 0xe)
			Gprecalc(1, 6, 11, 12, 0xc, 0xb)
			Gprecalc(2, 7, 8, 13, 0x8, 0x6)
			Gprecalc(3, 4, 9, 14, 0xd, 0x3)

			Gprecalc(0, 4, 8, 12, 0xc, 0x2)
			Gprecalc(1, 5, 9, 13, 0xa, 0x6)
			Gprecalc(2, 6, 10, 14, 0xb, 0x0)
			Gprecalc(3, 7, 11, 15, 0x3, 0x8)
			Gprecalc(0, 5, 10, 15, 0xd, 0x4)
			Gprecalc(1, 6, 11, 12, 0x5, 0x7)
			Gprecalc(2, 7, 8, 13, 0xe, 0xf)
			Gprecalc(3, 4, 9, 14, 0x9, 0x1)

			Gprecalc(0, 4, 8, 12, 0x5, 0xc)
			Gprecalc(1, 5, 9, 13, 0xf, 0x1)
			Gprecalc(2, 6, 10, 14, 0xd, 0xe)
			Gprecalc(3, 7, 11, 15, 0xa, 0x4)
			Gprecalc(0, 5, 10, 15, 0x7, 0x0)
			Gprecalc(1, 6, 11, 12, 0x3, 0x6)
			Gprecalc(2, 7, 8, 13, 0x2, 0x9)
			Gprecalc(3, 4, 9, 14, 0xb, 0x8)

			Gprecalc(0, 4, 8, 12, 0xb, 0xd)
			Gprecalc(1, 5, 9, 13, 0xe, 0x7)
			Gprecalc(2, 6, 10, 14, 0x1, 0xc)
			Gprecalc(3, 7, 11, 15, 0x9, 0x3)
			Gprecalc(0, 5, 10, 15, 0x0, 0x5)
			Gprecalc(1, 6, 11, 12, 0x4, 0xf)
			Gprecalc(2, 7, 8, 13, 0x6, 0x8)
			Gprecalc(3, 4, 9, 14, 0xa, 0x2)

			Gprecalc(0, 4, 8, 12, 0xf, 0x6)
			Gprecalc(1, 5, 9, 13, 0x9, 0xe)
			Gprecalc(2, 6, 10, 14, 0x3, 0xb)
			Gprecalc(3, 7, 11, 15, 0x8, 0x0)
			Gprecalc(0, 5, 10, 15, 0x2, 0xc)
			Gprecalc(1, 6, 11, 12, 0x7, 0xd)
			Gprecalc(2, 7, 8, 13, 0x4, 0x1)
			Gprecalc(3, 4, 9, 14, 0x5, 0xa)

			Gprecalc(0, 4, 8, 12, 0x2, 0xa)
			Gprecalc(1, 5, 9, 13, 0x4, 0x8)
			Gprecalc(2, 6, 10, 14, 0x6, 0x7)
			Gprecalc(3, 7, 11, 15, 0x5, 0x1)
			Gprecalc(0, 5, 10, 15, 0xb, 0xf)
			Gprecalc(1, 6, 11, 12, 0xe, 0x9)
			Gprecalc(2, 7, 8, 13, 0xc, 0x3)
			Gprecalc(3, 4, 9, 14, 0x0, 0xd)

			Gprecalc(0, 4, 8, 12, 0x1, 0x0)
			Gprecalc(1, 5, 9, 13, 0x3, 0x2)
			Gprecalc(2, 6, 10, 14, 0x5, 0x4)
			Gprecalc(3, 7, 11, 15, 0x7, 0x6)
			Gprecalc(0, 5, 10, 15, 0x9, 0x8)
			Gprecalc(1, 6, 11, 12, 0xb, 0xa)
			Gprecalc(2, 7, 8, 13, 0xd, 0xc)
			Gprecalc(3, 4, 9, 14, 0xf, 0xe)

			Gprecalc(0, 4, 8, 12, 0xa, 0xe)
			Gprecalc(1, 5, 9, 13, 0x8, 0x4)
			Gprecalc(2, 6, 10, 14, 0xf, 0x9)
			Gprecalc(3, 7, 11, 15, 0x6, 0xd)
			Gprecalc(0, 5, 10, 15, 0xc, 0x1)
			Gprecalc(1, 6, 11, 12, 0x2, 0x0)
			Gprecalc(2, 7, 8, 13, 0x7, 0xb)
			Gprecalc(3, 4, 9, 14, 0x3, 0x5)

			Gprecalc(0, 4, 8, 12, 0x8, 0xb)
			Gprecalc(1, 5, 9, 13, 0x0, 0xc)
			Gprecalc(2, 6, 10, 14, 0x2, 0x5)
			Gprecalc(3, 7, 11, 15, 0xd, 0xf)
			Gprecalc(0, 5, 10, 15, 0xe, 0xa)
			Gprecalc(1, 6, 11, 12, 0x6, 0x3)
			Gprecalc(2, 7, 8, 13, 0x1, 0x7)
			Gprecalc(3, 4, 9, 14, 0x4, 0x9)

			Gprecalc(0, 4, 8, 12, 0x9, 0x7)
			Gprecalc(1, 5, 9, 13, 0x1, 0x3)
			Gprecalc(2, 6, 10, 14, 0xc, 0xd)
			Gprecalc(3, 7, 11, 15, 0xe, 0xb)
			Gprecalc(0, 5, 10, 15, 0x6, 0x2)
			Gprecalc(1, 6, 11, 12, 0xa, 0x5)
			Gprecalc(2, 7, 8, 13, 0x0, 0x4)
			Gprecalc(3, 4, 9, 14, 0x8, 0xf)

			Gprecalc(0, 4, 8, 12, 0x0, 0x9)
			Gprecalc(1, 5, 9, 13, 0x7, 0x5)
			Gprecalc(2, 6, 10, 14, 0x4, 0x2)
			Gprecalc(3, 7, 11, 15, 0xf, 0xa)
			Gprecalc(0, 5, 10, 15, 0x1, 0xe)
			Gprecalc(1, 6, 11, 12, 0xc, 0xb)
			Gprecalc(2, 7, 8, 13, 0x8, 0x6)
			Gprecalc(3, 4, 9, 14, 0xd, 0x3)

			Gprecalc(0, 4, 8, 12, 0xc, 0x2)
			Gprecalc(1, 5, 9, 13, 0xa, 0x6)
			Gprecalc(2, 6, 10, 14, 0xb, 0x0)
			Gprecalc(3, 7, 11, 15, 0x3, 0x8)
			Gprecalc(0, 5, 10, 15, 0xd, 0x4)
			Gprecalc(1, 6, 11, 12, 0x5, 0x7)
			Gprecalc(2, 7, 8, 13, 0xe, 0xf)
			Gprecalc(3, 4, 9, 14, 0x9, 0x1)

			uint64_t *outHash = (uint64_t *)outputHash + 8 * thread;
		outHash[0] = devectorizeswap(h[0] ^ v[0] ^ v[8]);
		outHash[1] = devectorizeswap(h[1] ^ v[1] ^ v[9]);
		outHash[2] = devectorizeswap(h[2] ^ v[2] ^ v[10]);
		outHash[3] = devectorizeswap(h[3] ^ v[3] ^ v[11]);
		outHash[4] = devectorizeswap(h[4] ^ v[4] ^ v[12]);
		outHash[5] = devectorizeswap(h[5] ^ v[5] ^ v[13]);
		outHash[6] = devectorizeswap(h[6] ^ v[6] ^ v[14]);
		outHash[7] = devectorizeswap(h[7] ^ v[7] ^ v[15]);
	}
}



__global__ 
#if __CUDA_ARCH__ > 500
__launch_bounds__(32)
#else
__launch_bounds__(32, 16)
#endif
void quark_blake512_gpu_hash_80_multi(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ outputHash, const uint2*const __restrict__ c_PaddedMessage)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 block[16];
		const uint32_t nounce = startNounce + thread;

		block[0] = c_PaddedMessage[0];
		block[1] = c_PaddedMessage[1];
		block[2] = c_PaddedMessage[2];
		block[3] = c_PaddedMessage[3];
		block[4] = c_PaddedMessage[4];
		block[5] = c_PaddedMessage[5];
		block[6] = c_PaddedMessage[6];
		block[7] = c_PaddedMessage[7];
		block[8] = c_PaddedMessage[8];
		block[9] = c_PaddedMessage[9];
		block[10] = vectorizehigh(0x80000000);
		block[11] = vectorizelow(0);
		block[12] = vectorizelow(0);
		block[13] = vectorizelow(0x1);
		block[14] = vectorizelow(0);
		block[15] = vectorizelow(0x280);
		block[9].x = nounce;

		const uint2 u512[16] =
		{
			{ 0x85a308d3UL, 0x243f6a88 }, { 0x03707344UL, 0x13198a2e },
			{ 0x299f31d0UL, 0xa4093822 }, { 0xec4e6c89UL, 0x082efa98 },
			{ 0x38d01377UL, 0x452821e6 }, { 0x34e90c6cUL, 0xbe5466cf },
			{ 0xc97c50ddUL, 0xc0ac29b7 }, { 0xb5470917UL, 0x3f84d5b5 },
			{ 0x8979fb1bUL, 0x9216d5d9 }, { 0x98dfb5acUL, 0xd1310ba6 },
			{ 0xd01adfb7UL, 0x2ffd72db }, { 0x6a267e96UL, 0xb8e1afed },
			{ 0xf12c7f99UL, 0xba7c9045 }, { 0xb3916cf7UL, 0x24a19947 },
			{ 0x858efc16UL, 0x0801f2e2 }, { 0x71574e69UL, 0x636920d8 }
		};

		const uint2 h[8] = {
				{ 0xf3bcc908UL,0x6a09e667UL },
				{ 0x84caa73bUL ,0xbb67ae85UL },
				{ 0xfe94f82bUL,0x3c6ef372UL },
				{ 0x5f1d36f1UL,0xa54ff53aUL },
				{ 0xade682d1UL,0x510e527fUL },
				{ 0x2b3e6c1fUL,0x9b05688cUL },
				{ 0xfb41bd6bUL,0x1f83d9abUL },
				{ 0x137e2179UL,0x5be0cd19UL }
		};

		uint2 v[16] =
		{
			h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
			u512[0], u512[1], u512[2], u512[3], u512[4] ^ 640, u512[5] ^ 640, u512[6], u512[7]
		};

		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
		Gprecalc(1, 5, 9, 13, 0x3, 0x2)
		Gprecalc(2, 6, 10, 14, 0x5, 0x4)
		Gprecalc(3, 7, 11, 15, 0x7, 0x6)
		Gprecalc(0, 5, 10, 15, 0x9, 0x8)
		Gprecalc(1, 6, 11, 12, 0xb, 0xa)
		Gprecalc(2, 7, 8, 13, 0xd, 0xc)
		Gprecalc(3, 4, 9, 14, 0xf, 0xe)

		Gprecalc(0, 4, 8, 12, 0xa, 0xe)
		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		Gprecalc(0, 5, 10, 15, 0xc, 0x1)
		Gprecalc(1, 6, 11, 12, 0x2, 0x0)
		Gprecalc(2, 7, 8, 13, 0x7, 0xb)
		Gprecalc(3, 4, 9, 14, 0x3, 0x5)

		Gprecalc(0, 4, 8, 12, 0x8, 0xb)
		Gprecalc(1, 5, 9, 13, 0x0, 0xc)
		Gprecalc(2, 6, 10, 14, 0x2, 0x5)
		Gprecalc(3, 7, 11, 15, 0xd, 0xf)
		Gprecalc(0, 5, 10, 15, 0xe, 0xa)
		Gprecalc(1, 6, 11, 12, 0x6, 0x3)
		Gprecalc(2, 7, 8, 13, 0x1, 0x7)
		Gprecalc(3, 4, 9, 14, 0x4, 0x9)

		Gprecalc(0, 4, 8, 12, 0x9, 0x7)
		Gprecalc(1, 5, 9, 13, 0x1, 0x3)
		Gprecalc(2, 6, 10, 14, 0xc, 0xd)
		Gprecalc(3, 7, 11, 15, 0xe, 0xb)
		Gprecalc(0, 5, 10, 15, 0x6, 0x2)
		Gprecalc(1, 6, 11, 12, 0xa, 0x5)
		Gprecalc(2, 7, 8, 13, 0x0, 0x4)
		Gprecalc(3, 4, 9, 14, 0x8, 0xf)

		Gprecalc(0, 4, 8, 12, 0x0, 0x9)
		Gprecalc(1, 5, 9, 13, 0x7, 0x5)
		Gprecalc(2, 6, 10, 14, 0x4, 0x2)
		Gprecalc(3, 7, 11, 15, 0xf, 0xa)
		Gprecalc(0, 5, 10, 15, 0x1, 0xe)
		Gprecalc(1, 6, 11, 12, 0xc, 0xb)
		Gprecalc(2, 7, 8, 13, 0x8, 0x6)
		Gprecalc(3, 4, 9, 14, 0xd, 0x3)
		
		Gprecalc(0, 4, 8, 12, 0xc, 0x2)
		Gprecalc(1, 5, 9, 13, 0xa, 0x6)
		Gprecalc(2, 6, 10, 14, 0xb, 0x0)
		Gprecalc(3, 7, 11, 15, 0x3, 0x8)
		Gprecalc(0, 5, 10, 15, 0xd, 0x4)
		Gprecalc(1, 6, 11, 12, 0x5, 0x7)
		Gprecalc(2, 7, 8, 13, 0xe, 0xf)
		Gprecalc(3, 4, 9, 14, 0x9, 0x1)

		Gprecalc(0, 4, 8, 12, 0x5, 0xc)
		Gprecalc(1, 5, 9, 13, 0xf, 0x1)
		Gprecalc(2, 6, 10, 14, 0xd, 0xe)
		Gprecalc(3, 7, 11, 15, 0xa, 0x4)
		Gprecalc(0, 5, 10, 15, 0x7, 0x0)
		Gprecalc(1, 6, 11, 12, 0x3, 0x6)
		Gprecalc(2, 7, 8, 13, 0x2, 0x9)
		Gprecalc(3, 4, 9, 14, 0xb, 0x8)

		Gprecalc(0, 4, 8, 12, 0xb, 0xd)
		Gprecalc(1, 5, 9, 13, 0xe, 0x7)
		Gprecalc(2, 6, 10, 14, 0x1, 0xc)
		Gprecalc(3, 7, 11, 15, 0x9, 0x3)
		Gprecalc(0, 5, 10, 15, 0x0, 0x5)
		Gprecalc(1, 6, 11, 12, 0x4, 0xf)
		Gprecalc(2, 7, 8, 13, 0x6, 0x8)
		Gprecalc(3, 4, 9, 14, 0xa, 0x2)

		Gprecalc(0, 4, 8, 12, 0xf, 0x6)
		Gprecalc(1, 5, 9, 13, 0x9, 0xe)
		Gprecalc(2, 6, 10, 14, 0x3, 0xb)
		Gprecalc(3, 7, 11, 15, 0x8, 0x0)
		Gprecalc(0, 5, 10, 15, 0x2, 0xc)
		Gprecalc(1, 6, 11, 12, 0x7, 0xd)
		Gprecalc(2, 7, 8, 13, 0x4, 0x1)
		Gprecalc(3, 4, 9, 14, 0x5, 0xa)

		Gprecalc(0, 4, 8, 12, 0x2, 0xa)
		Gprecalc(1, 5, 9, 13, 0x4, 0x8)
		Gprecalc(2, 6, 10, 14, 0x6, 0x7)
		Gprecalc(3, 7, 11, 15, 0x5, 0x1)
		Gprecalc(0, 5, 10, 15, 0xb, 0xf)
		Gprecalc(1, 6, 11, 12, 0xe, 0x9)
		Gprecalc(2, 7, 8, 13, 0xc, 0x3)
		Gprecalc(3, 4, 9, 14, 0x0, 0xd)
		
		Gprecalc(0, 4, 8, 12, 0x1, 0x0)
		Gprecalc(1, 5, 9, 13, 0x3, 0x2)
		Gprecalc(2, 6, 10, 14, 0x5, 0x4)
		Gprecalc(3, 7, 11, 15, 0x7, 0x6)
		Gprecalc(0, 5, 10, 15, 0x9, 0x8)
		Gprecalc(1, 6, 11, 12, 0xb, 0xa)
		Gprecalc(2, 7, 8, 13, 0xd, 0xc)
		Gprecalc(3, 4, 9, 14, 0xf, 0xe)

		Gprecalc(0, 4, 8, 12, 0xa, 0xe)
		Gprecalc(1, 5, 9, 13, 0x8, 0x4)
		Gprecalc(2, 6, 10, 14, 0xf, 0x9)
		Gprecalc(3, 7, 11, 15, 0x6, 0xd)
		Gprecalc(0, 5, 10, 15, 0xc, 0x1)
		Gprecalc(1, 6, 11, 12, 0x2, 0x0)
		Gprecalc(2, 7, 8, 13, 0x7, 0xb)
		Gprecalc(3, 4, 9, 14, 0x3, 0x5)

		Gprecalc(0, 4, 8, 12, 0x8, 0xb)
		Gprecalc(1, 5, 9, 13, 0x0, 0xc)
		Gprecalc(2, 6, 10, 14, 0x2, 0x5)
		Gprecalc(3, 7, 11, 15, 0xd, 0xf)
		Gprecalc(0, 5, 10, 15, 0xe, 0xa)
		Gprecalc(1, 6, 11, 12, 0x6, 0x3)
		Gprecalc(2, 7, 8, 13, 0x1, 0x7)
		Gprecalc(3, 4, 9, 14, 0x4, 0x9)

		Gprecalc(0, 4, 8, 12, 0x9, 0x7)
		Gprecalc(1, 5, 9, 13, 0x1, 0x3)
		Gprecalc(2, 6, 10, 14, 0xc, 0xd)
		Gprecalc(3, 7, 11, 15, 0xe, 0xb)
		Gprecalc(0, 5, 10, 15, 0x6, 0x2)
		Gprecalc(1, 6, 11, 12, 0xa, 0x5)
		Gprecalc(2, 7, 8, 13, 0x0, 0x4)
		Gprecalc(3, 4, 9, 14, 0x8, 0xf)

		Gprecalc(0, 4, 8, 12, 0x0, 0x9)
		Gprecalc(1, 5, 9, 13, 0x7, 0x5)
		Gprecalc(2, 6, 10, 14, 0x4, 0x2)
		Gprecalc(3, 7, 11, 15, 0xf, 0xa)
		Gprecalc(0, 5, 10, 15, 0x1, 0xe)
		Gprecalc(1, 6, 11, 12, 0xc, 0xb)
		Gprecalc(2, 7, 8, 13, 0x8, 0x6)
		Gprecalc(3, 4, 9, 14, 0xd, 0x3)
		
		Gprecalc(0, 4, 8, 12, 0xc, 0x2)
		Gprecalc(1, 5, 9, 13, 0xa, 0x6)
		Gprecalc(2, 6, 10, 14, 0xb, 0x0)
		Gprecalc(3, 7, 11, 15, 0x3, 0x8)
		Gprecalc(0, 5, 10, 15, 0xd, 0x4)
		Gprecalc(1, 6, 11, 12, 0x5, 0x7)
		Gprecalc(2, 7, 8, 13, 0xe, 0xf)
		Gprecalc(3, 4, 9, 14, 0x9, 0x1)

		uint64_t *outHash = (uint64_t *)outputHash + 8 * thread;
		outHash[0] = devectorizeswap(h[0] ^ v[0] ^ v[8]);
		outHash[1] = devectorizeswap(h[1] ^ v[1] ^ v[9]);
		outHash[2] = devectorizeswap(h[2] ^ v[2] ^ v[10]);
		outHash[3] = devectorizeswap(h[3] ^ v[3] ^ v[11]);
		outHash[4] = devectorizeswap(h[4] ^ v[4] ^ v[12]);
		outHash[5] = devectorizeswap(h[5] ^ v[5] ^ v[13]);
		outHash[6] = devectorizeswap(h[6] ^ v[6] ^ v[14]);
		outHash[7] = devectorizeswap(h[7] ^ v[7] ^ v[15]);
	}
}


// ---------------------------- END CUDA quark_blake512 functions ------------------------------------

__host__ void quark_blake512_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&c_PaddedMessage80[thr_id], 10 * sizeof(uint2)));
}

__host__ void quark_blake512_cpu_setBlock_80_multi(uint32_t thr_id, uint64_t *pdata)
{
	uint64_t PaddedMessage[10];
	for (int i = 0; i < 10; i++)
		PaddedMessage[i] = cuda_swab64(pdata[i]);
	CUDA_SAFE_CALL(cudaMemcpy(c_PaddedMessage80[thr_id], PaddedMessage, 10 * sizeof(uint64_t), cudaMemcpyHostToDevice));
}

__host__ void quark_blake512_cpu_setBlock_80(uint64_t *pdata)
{
	uint64_t PaddedMessage[10];
	for (int i = 0; i < 10; i++)
		PaddedMessage[i] = cuda_swab64(pdata[i]);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedM, PaddedMessage, 10 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));

}


__host__ void quark_blake512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash)
{
	const uint32_t threadsperblock = 32;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	quark_blake512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_nonceVector, (uint64_t*)d_outputHash);
}

__host__ void quark_blake512_cpu_hash_80_multi(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{

	const uint32_t threadsperblock = 32;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	quark_blake512_gpu_hash_80_multi << <grid, block >> >(threads, startNounce, d_outputHash, c_PaddedMessage80[thr_id]);
}
__host__ void quark_blake512_cpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{

	const uint32_t threadsperblock = 32;
	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	quark_blake512_gpu_hash_80 << <grid, block >> >(threads, startNounce, d_outputHash);
}
