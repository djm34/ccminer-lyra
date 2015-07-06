/*
 * Quick and dirty addition of Fugue-512 for X13
 * 
 * Built on cbuchner1's implementation, actual hashing code
 * heavily based on phm's sgminer
 *
 */
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"


__constant__ uint32_t pTarget[8];
static uint32_t *d_nonce[MAX_GPUS];

/*
 * X13 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  phm
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   phm <phm@inbox.com>
 */

__constant__ uint32_t mixTab0Tex[] = {
	(0x63633297), (0x7c7c6feb), (0x77775ec7),
	(0x7b7b7af7), (0xf2f2e8e5), (0x6b6b0ab7),
	(0x6f6f16a7), (0xc5c56d39), (0x303090c0),
	(0x01010704), (0x67672e87), (0x2b2bd1ac),
	(0xfefeccd5), (0xd7d71371), (0xabab7c9a),
	(0x767659c3), (0xcaca4005), (0x8282a33e),
	(0xc9c94909), (0x7d7d68ef), (0xfafad0c5),
	(0x5959947f), (0x4747ce07), (0xf0f0e6ed),
	(0xadad6e82), (0xd4d41a7d), (0xa2a243be),
	(0xafaf608a), (0x9c9cf946), (0xa4a451a6),
	(0x727245d3), (0xc0c0762d), (0xb7b728ea),
	(0xfdfdc5d9), (0x9393d47a), (0x2626f298),
	(0x363682d8), (0x3f3fbdfc), (0xf7f7f3f1),
	(0xcccc521d), (0x34348cd0), (0xa5a556a2),
	(0xe5e58db9), (0xf1f1e1e9), (0x71714cdf),
	(0xd8d83e4d), (0x313197c4), (0x15156b54),
	(0x04041c10), (0xc7c76331), (0x2323e98c),
	(0xc3c37f21), (0x18184860), (0x9696cf6e),
	(0x05051b14), (0x9a9aeb5e), (0x0707151c),
	(0x12127e48), (0x8080ad36), (0xe2e298a5),
	(0xebeba781), (0x2727f59c), (0xb2b233fe),
	(0x757550cf), (0x09093f24), (0x8383a43a),
	(0x2c2cc4b0), (0x1a1a4668), (0x1b1b416c),
	(0x6e6e11a3), (0x5a5a9d73), (0xa0a04db6),
	(0x5252a553), (0x3b3ba1ec), (0xd6d61475),
	(0xb3b334fa), (0x2929dfa4), (0xe3e39fa1),
	(0x2f2fcdbc), (0x8484b126), (0x5353a257),
	(0xd1d10169), (0x00000000), (0xededb599),
	(0x2020e080), (0xfcfcc2dd), (0xb1b13af2),
	(0x5b5b9a77), (0x6a6a0db3), (0xcbcb4701),
	(0xbebe17ce), (0x3939afe4), (0x4a4aed33),
	(0x4c4cff2b), (0x5858937b), (0xcfcf5b11),
	(0xd0d0066d), (0xefefbb91), (0xaaaa7b9e),
	(0xfbfbd7c1), (0x4343d217), (0x4d4df82f),
	(0x333399cc), (0x8585b622), (0x4545c00f),
	(0xf9f9d9c9), (0x02020e08), (0x7f7f66e7),
	(0x5050ab5b), (0x3c3cb4f0), (0x9f9ff04a),
	(0xa8a87596), (0x5151ac5f), (0xa3a344ba),
	(0x4040db1b), (0x8f8f800a), (0x9292d37e),
	(0x9d9dfe42), (0x3838a8e0), (0xf5f5fdf9),
	(0xbcbc19c6), (0xb6b62fee), (0xdada3045),
	(0x2121e784), (0x10107040), (0xffffcbd1),
	(0xf3f3efe1), (0xd2d20865), (0xcdcd5519),
	(0x0c0c2430), (0x1313794c), (0xececb29d),
	(0x5f5f8667), (0x9797c86a), (0x4444c70b),
	(0x1717655c), (0xc4c46a3d), (0xa7a758aa),
	(0x7e7e61e3), (0x3d3db3f4), (0x6464278b),
	(0x5d5d886f), (0x19194f64), (0x737342d7),
	(0x60603b9b), (0x8181aa32), (0x4f4ff627),
	(0xdcdc225d), (0x2222ee88), (0x2a2ad6a8),
	(0x9090dd76), (0x88889516), (0x4646c903),
	(0xeeeebc95), (0xb8b805d6), (0x14146c50),
	(0xdede2c55), (0x5e5e8163), (0x0b0b312c),
	(0xdbdb3741), (0xe0e096ad), (0x32329ec8),
	(0x3a3aa6e8), (0x0a0a3628), (0x4949e43f),
	(0x06061218), (0x2424fc90), (0x5c5c8f6b),
	(0xc2c27825), (0xd3d30f61), (0xacac6986),
	(0x62623593), (0x9191da72), (0x9595c662),
	(0xe4e48abd), (0x797974ff), (0xe7e783b1),
	(0xc8c84e0d), (0x373785dc), (0x6d6d18af),
	(0x8d8d8e02), (0xd5d51d79), (0x4e4ef123),
	(0xa9a97292), (0x6c6c1fab), (0x5656b943),
	(0xf4f4fafd), (0xeaeaa085), (0x6565208f),
	(0x7a7a7df3), (0xaeae678e), (0x08083820),
	(0xbaba0bde), (0x787873fb), (0x2525fb94),
	(0x2e2ecab8), (0x1c1c5470), (0xa6a65fae),
	(0xb4b421e6), (0xc6c66435), (0xe8e8ae8d),
	(0xdddd2559), (0x747457cb), (0x1f1f5d7c),
	(0x4b4bea37), (0xbdbd1ec2), (0x8b8b9c1a),
	(0x8a8a9b1e), (0x70704bdb), (0x3e3ebaf8),
	(0xb5b526e2), (0x66662983), (0x4848e33b),
	(0x0303090c), (0xf6f6f4f5), (0x0e0e2a38),
	(0x61613c9f), (0x35358bd4), (0x5757be47),
	(0xb9b902d2), (0x8686bf2e), (0xc1c17129),
	(0x1d1d5374), (0x9e9ef74e), (0xe1e191a9),
	(0xf8f8decd), (0x9898e556), (0x11117744),
	(0x696904bf), (0xd9d93949), (0x8e8e870e),
	(0x9494c166), (0x9b9bec5a), (0x1e1e5a78),
	(0x8787b82a), (0xe9e9a989), (0xcece5c15),
	(0x5555b04f), (0x2828d8a0), (0xdfdf2b51),
	(0x8c8c8906), (0xa1a14ab2), (0x89899212),
	(0x0d0d2334), (0xbfbf10ca), (0xe6e684b5),
	(0x4242d513), (0x686803bb), (0x4141dc1f),
	(0x9999e252), (0x2d2dc3b4), (0x0f0f2d3c),
	(0xb0b03df6), (0x5454b74b), (0xbbbb0cda),
	(0x16166258)
};

#define TIX4(q, x00, x01, x04, x07, x08, x22, x24, x27, x30) { \
		x22 ^= x00; \
		x00 = (q); \
		x08 ^= x00; \
		x01 ^= x24; \
		x04 ^= x27; \
		x07 ^= x30; \
	}

#define CMIX36(x00, x01, x02, x04, x05, x06, x18, x19, x20) { \
		x00 ^= x04; \
		x01 ^= x05; \
		x02 ^= x06; \
		x18 ^= x04; \
		x19 ^= x05; \
		x20 ^= x06; \
	}
#define SMIX(x0, x1, x2, x3) { \
		uint32_t tmp = mixtabs[__byte_perm(x0, 0, 0x4443)]; \
		uint32_t c0 = tmp; \
		tmp = mixtabs[256 + __byte_perm(x0, 0, 0x4442)]; \
		c0 ^= tmp; \
		uint32_t r1 = tmp; \
		tmp = mixtabs[512 + __byte_perm(x0, 0, 0x4441)]; \
		c0 ^= tmp; \
		uint32_t r2= tmp; \
		tmp = mixtabs[768+(x0 & 0xff)]; \
		c0 ^= tmp; \
		uint32_t r3= tmp; \
		tmp = mixtabs[__byte_perm(x1, 0, 0x4443)]; \
		uint32_t c1 = tmp; \
		uint32_t r0 = tmp; \
		tmp = mixtabs[256 +__byte_perm(x1, 0, 0x4442)]; \
		c1 ^= tmp; \
		tmp = mixtabs[512 +__byte_perm(x1, 0, 0x4441)]; \
		c1 ^= tmp; \
		r2 ^= tmp; \
		tmp = mixtabs[768 +(x1 & 0xff)]; \
		c1 ^= tmp; \
		r3 ^= tmp; \
		tmp = mixtabs[__byte_perm(x2, 0, 0x4443)]; \
		uint32_t c2 = tmp; \
		r0 ^= tmp; \
		tmp = mixtabs[256 +__byte_perm(x2, 0, 0x4442)]; \
		c2 ^= tmp; \
		r1 ^= tmp; \
		tmp = mixtabs[512 +__byte_perm(x2, 0, 0x4441)]; \
		c2 ^= tmp; \
		tmp = mixtabs[768 +(x2 & 0xff)]; \
		c2 ^= tmp; \
		r3 ^= tmp; \
		tmp = mixtabs[__byte_perm(x3, 0, 0x4443)]; \
		uint32_t c3 = tmp; \
		r0 ^= tmp; \
		tmp = mixtabs[256 +__byte_perm(x3, 0, 0x4442)]; \
		c3 ^= tmp; \
		r1 ^= tmp; \
		tmp = mixtabs[512 +__byte_perm(x3, 0, 0x4441)]; \
		c3 ^= tmp; \
		r2 ^= tmp; \
		tmp = mixtabs[768 +(x3 & 0xff)]; \
		c3 ^= tmp; \
		uint32_t tmp2 = __byte_perm((c0 ^ r0),(c1 ^ r1), 0x3636);\
		tmp= __byte_perm((c2 ^ r2),(c3 ^ r3), 0x1414); \
		x0 = __byte_perm(tmp2,tmp, 0x3254);\
		r0 = ROL8(r0); \
		r1 = ROL8(r1); \
		r2 = ROL8(r2); \
		r3 = ROL8(r3); \
		tmp2 = __byte_perm((c1 ^ r0),(c2 ^ r1), 0x3636);\
		tmp= __byte_perm((c3 ^ r2),(c0 ^ r3), 0x1414); \
		x1 = __byte_perm(tmp2,tmp, 0x3254);\
		r0 = ROL8(r0); \
		r1 = ROL8(r1); \
		r2 = ROL8(r2); \
		r3 = ROL8(r3); \
		tmp2 = __byte_perm((c2 ^ r0),(c3 ^ r1), 0x3636);\
		tmp= __byte_perm((c0 ^ r2),(c1 ^ r3), 0x1414); \
		x2 = __byte_perm(tmp2,tmp, 0x3254);\
		r0 = ROL8(r0); \
		r1 = ROL8(r1); \
		r2 = ROL8(r2); \
		r3 = ROL8(r3); \
		tmp2 = __byte_perm((c3 ^ r0),(c0 ^ r1), 0x3636);\
		tmp= __byte_perm((c1 ^ r2),(c2 ^ r3), 0x1414); \
		x3 = __byte_perm(tmp2,tmp, 0x3254);\
		}
#define SMIX0(x0, x1, x2, x3) { \
		uint32_t tmp = mixtabs[__byte_perm(x0, 0, 0x4443)]; \
		uint32_t c0 = tmp; \
		tmp = mixtabs[256 +__byte_perm(x0, 0, 0x4442)]; \
		c0 ^= tmp; \
		uint32_t r1 = tmp; \
		tmp = mixtabs[512 +__byte_perm(x0, 0, 0x4441)]; \
		c0 ^= tmp; \
		uint32_t r2= tmp; \
		tmp = mixtabs[768 +(x0 & 0xff)]; \
		c0 ^= tmp; \
		uint32_t r3= tmp; \
		tmp = mixtabs[__byte_perm(x1, 0, 0x4443)]; \
		uint32_t c1 = tmp; \
		uint32_t r0 = tmp; \
		tmp = mixtabs[256 +__byte_perm(x1, 0, 0x4442)]; \
		c1 ^= tmp; \
		tmp = mixtabs[512 +__byte_perm(x1, 0, 0x4441)]; \
		c1 ^= tmp; \
		r2 ^= tmp; \
		tmp = mixtabs[768 +(x1 & 0xff)]; \
		c1 ^= tmp; \
		r3 ^= tmp; \
		tmp = mixtabs[__byte_perm(x2, 0, 0x4443)]; \
		uint32_t c2 = tmp; \
		r0 ^= tmp; \
		tmp = mixtabs[256 +__byte_perm(x2, 0, 0x4442)]; \
		c2 ^= tmp; \
		r1 ^= tmp; \
		tmp = mixtabs[512 +__byte_perm(x2, 0, 0x4441)]; \
		c2 ^= tmp; \
		tmp = mixtabs[768 +(x2 & 0xff)]; \
		c2 ^= tmp; \
		r3 ^= tmp; \
		tmp = mixtabs[__byte_perm(x3, 0, 0x4443)]; \
		uint32_t c3 = tmp; \
		r0 ^= tmp; \
		tmp = mixtabs[256 +__byte_perm(x3, 0, 0x4442)]; \
		c3 ^= tmp; \
		r1 ^= tmp; \
		tmp = mixtabs[512 +__byte_perm(x3, 0, 0x4441)]; \
		c3 ^= tmp; \
		r2 ^= tmp; \
		tmp = mixtabs[768 +(x3 & 0xff)]; \
		c3 ^= tmp; \
		uint32_t tmp2 = __byte_perm((c0 ^ r0),(c1 ^ r1), 0x3636);\
		tmp= __byte_perm((c2 ^ r2),(c3 ^ r3), 0x1414); \
		x0 = __byte_perm(tmp2,tmp, 0x3254);\
		}

#define ROR3 { \
	B33 = S33, B34 = S34, B35 = S35; \
    S35 = S32; S34 = S31; S33 = S30; S32 = S29; S31 = S28; S30 = S27; S29 = S26; S28 = S25; S27 = S24; \
	S26 = S23; S25 = S22; S24 = S21; S23 = S20; S22 = S19; S21 = S18; S20 = S17; S19 = S16; S18 = S15; \
	S17 = S14; S16 = S13; S15 = S12; S14 = S11; S13 = S10; S12 = S09; S11 = S08; S10 = S07; S09 = S06; \
	S08 = S05; S07 = S04; S06 = S03; S05 = S02; S04 = S01; S03 = S00; S02 = B35; S01 = B34; S00 = B33; \
	}

#define ROL1 { \
			B35 = S00; \
			S00 = S01; S01 = S02; S02 = S03; S03 = S04; S04 = S05; S05 = S06; S06 = S07; S07 = S08; S08 = S09; S09 = S10; \
			S10 = S11; S11 = S12; S12 = S13; S13 = S14; S14 = S15; S15 = S16; S16 = S17; S17 = S18; S18 = S19; S19 = S20; \
			S20 = S21; S21 = S22; S22 = S23; S23 = S24; S24 = S25; S25 = S26; S26 = S27; S27 = S28; S28 = S29; S29 = S30; \
			S30 = S31; S31 = S32; S32 = S33; S33 = S34; S34 = S35; \
			S35 = B35; \
	}

#define FUGUE512_3(x, y, z) {  \
        TIX4(x, S00, S01, S04, S07, S08, S22, S24, S27, S30); \
        CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17); \
        SMIX(S33, S34, S35, S00); \
        CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14); \
        SMIX(S30, S31, S32, S33); \
        CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11); \
        SMIX(S27, S28, S29, S30); \
        CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08); \
        SMIX(S24, S25, S26, S27); \
        \
        TIX4(y, S24, S25, S28, S31, S32, S10, S12, S15, S18); \
        CMIX36(S21, S22, S23, S25, S26, S27, S03, S04, S05); \
        SMIX(S21, S22, S23, S24); \
        CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02); \
        SMIX(S18, S19, S20, S21); \
        CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35); \
        SMIX(S15, S16, S17, S18); \
        CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32); \
        SMIX(S12, S13, S14, S15); \
        \
        TIX4(z, S12, S13, S16, S19, S20, S34, S00, S03, S06); \
        CMIX36(S09, S10, S11, S13, S14, S15, S27, S28, S29); \
        SMIX(S09, S10, S11, S12); \
        CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26); \
        SMIX(S06, S07, S08, S09); \
        CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23); \
        SMIX(S03, S04, S05, S06); \
        CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20); \
        SMIX(S00, S01, S02, S03); \
	}

__global__ __launch_bounds__(128, 8)
void x13_fugue512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
	__shared__ uint32_t mixtabs[1024];

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		if (threadIdx.x < 128)
		{
			mixtabs[threadIdx.x] = mixTab0Tex[threadIdx.x];
			mixtabs[threadIdx.x + 128] = mixTab0Tex[threadIdx.x + 128];
			mixtabs[256 + threadIdx.x] = ROL24(mixtabs[threadIdx.x]);
			mixtabs[256 + threadIdx.x + 128] = ROL24(mixtabs[threadIdx.x + 128]);
			mixtabs[512 + threadIdx.x] = ROL16(mixtabs[threadIdx.x]);
			mixtabs[512 + threadIdx.x + 128] = ROL16(mixtabs[threadIdx.x + 128]);
			mixtabs[768 + threadIdx.x] = ROL8(mixtabs[threadIdx.x]);
			mixtabs[768 + threadIdx.x + 128] = ROL8(mixtabs[threadIdx.x + 128]);
		}
		const uint32_t nounce =  (startNounce + thread);

		const int hashPosition = nounce - startNounce;
		uint32_t *const Hash = &g_hash[hashPosition*16];

#pragma unroll 16
		for (int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(Hash[i]);

		uint32_t S00, S01, S02, S03, S04, S05, S06, S07, S08, S09;
		uint32_t S10, S11, S12, S13, S14, S15, S16, S17, S18, S19;
		uint32_t S20, S21, S22, S23, S24, S25, S26, S27, S28, S29;
		uint32_t S30, S31, S32, S33, S34, S35;

		uint32_t B33, B34, B35;

		S02 = S03 = S05 = S06 = S09 = S10 = S11 = S12 = S13 = S14 = S16 = S17 = S18 = S19 = 0;
		S20 = 0x8807a57eUL; S21 = 0xe616af75UL; S22 = 0xc5d3e4dbUL; S23 = 0xac9ab027UL;
		S24 = 0xd915f117UL; S25 = 0xb6eecc54UL; S26 = 0x06e8020bUL; S27 = 0x4a92efd1UL;
		S28 = 0xaac6e2c9UL; S29 = 0xddb21398UL; S30 = 0xcae65838UL; S31 = 0x437f203fUL;
		S32 = 0x25ea78e7UL; S33 = 0x4c0a2cc1UL; S34 = 0xda6ed11dUL; S35 = 0xe13e3567UL;

		S01 = 0xd915f117UL;
		S04 = 0x4a92efd1UL;
		S07 = 0xcae65838UL;
		S15 = 0xd915f117UL;
		S00 = Hash[0];
		S08 = Hash[0];

		uint32_t c0 = 0x9ae23283UL;
		uint32_t c1 = 0x0361b92dUL;
		uint32_t c2 = 0x4c92d8edUL;
		uint32_t r0, r1, r2;
		uint32_t tmp, tmp2, c3;

		tmp = mixtabs[__byte_perm(S00, 0, 17475)]; c3 = tmp; r0 = 0xafaf608aUL ^ tmp;
		tmp = mixtabs[256 + __byte_perm(S00, 0, 17474)]; c3 ^= tmp; r1 = 0x79d5d51dUL ^ tmp;
		tmp = mixtabs[512 + __byte_perm(S00, 0, 17473)]; c3 ^= tmp; r2 = 0xf6274f4fUL ^ tmp;
		tmp = mixtabs[768 + __byte_perm(S00, 0, 17472)]; c3 ^= tmp;
		tmp2 = __byte_perm(c0 ^ r0, c1 ^ r1, 13878);
		tmp = __byte_perm(c2 ^ r2, c3 ^ 0x59947f59UL, 5140);
		S33 = __byte_perm(tmp2, tmp, 12884);
		r0 = ROL8(r0); r1 = ROL8(r1);
		r2 = ROL8(r2);
		tmp2 = __byte_perm(c1 ^ r0, c2 ^ r1, 13878);
		tmp = __byte_perm(c3 ^ r2, c0 ^ 0x947f5959UL, 5140);
		S34 = __byte_perm(tmp2, tmp, 12884);
		r0 = ROL8(r0); r1 = ROL8(r1);
		r2 = ROL8(r2);
		tmp2 = __byte_perm(c2 ^ r0, c3 ^ r1, 13878);
		tmp = __byte_perm(c0 ^ r2, c1 ^ 0x7f595994UL, 5140);
		S35 = __byte_perm(tmp2, tmp, 12884);
		r0 = ROL8(r0); r1 = ROL8(r1);
		r2 = ROL8(r2);
		tmp2 = __byte_perm(c3 ^ r0, c0 ^ r1, 13878);
		tmp = __byte_perm(c1 ^ r2, c2 ^ 0x5959947fUL, 5140);
		S00 = __byte_perm(tmp2, tmp, 12884);

		CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14);
		SMIX(S30, S31, S32, S33);
		CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11);
		SMIX(S27, S28, S29, S30);
		CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08);
		SMIX(S24, S25, S26, S27);

		TIX4(Hash[1], S24, S25, S28, S31, S32, S10, S12, S15, S18);
		CMIX36(S21, S22, S23, S25, S26, S27, S03, S04, S05);
		SMIX(S21, S22, S23, S24);
		CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02);
		SMIX(S18, S19, S20, S21);
		CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35);
		SMIX(S15, S16, S17, S18);
		CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32);
		SMIX(S12, S13, S14, S15);

		TIX4(Hash[2], S12, S13, S16, S19, S20, S34, S00, S03, S06);
		CMIX36(S09, S10, S11, S13, S14, S15, S27, S28, S29);
		SMIX(S09, S10, S11, S12);
		CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26);
		SMIX(S06, S07, S08, S09);
		CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23);
		SMIX(S03, S04, S05, S06);
		CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
		SMIX(S00, S01, S02, S03);
#pragma unroll
		for (int i = 3; i < (5 * 3); i += 3)
		{
			FUGUE512_3((Hash[i]), (Hash[i + 1]), (Hash[i + 2]));
		}
		TIX4(Hash[0xF], S00, S01, S04, S07, S08, S22, S24, S27, S30);
		CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17);
		SMIX(S33, S34, S35, S00);
		CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14);
		SMIX(S30, S31, S32, S33);
		CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11);
		SMIX(S27, S28, S29, S30);
		CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08);
		SMIX(S24, S25, S26, S27);

		S10 ^= S24;
		S25 ^= S12; S28 ^= S15; S31 ^= S18;
		S21 ^= S25; S22 ^= S26; S23 ^= S27; S03 ^= S25; S04 ^= S26; S05 ^= S27;
		tmp = (*(mixtabs + ((__byte_perm(S21, 0, 0x4443))))); c0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S21, 0, 0x4442))))); c0 ^= tmp; r1 = tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S21, 0, 0x4441))))); c0 ^= tmp; r2 = tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S21, 0, 0x4440))))); c0 ^= tmp; uint32_t r3 = tmp; tmp = (*(mixtabs + ((__byte_perm(S22, 0, 0x4443))))); c1 = tmp; r0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S22, 0, 0x4442))))); c1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S22, 0, 0x4441))))); c1 ^= tmp; r2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S22, 0, 0x4440))))); c1 ^= tmp; r3 ^= tmp; tmp = (*(mixtabs + ((__byte_perm(S23, 0, 0x4443))))); c2 = tmp; r0 ^= tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S23, 0, 0x4442))))); c2 ^= tmp; r1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S23, 0, 0x4441))))); c2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S23, 0, 0x4440))))); c2 ^= tmp; r3 ^= tmp;
		r0 ^= 0x63633297UL;
		r1 ^= 0x97636332UL;
		r2 ^= 0x32976363UL;
		c3 = (0x63633297UL ^ 0x97636332UL ^ 0x32976363UL ^ 0x63329763UL);
		tmp2 = __byte_perm((c0 ^ r0), (c1 ^ r1), 0x3636); tmp = __byte_perm((c2 ^ r2), (c3 ^ r3), 0x1414); S21 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c1 ^ r0), (c2 ^ r1), 0x3636); tmp = __byte_perm((c3 ^ r2), (c0 ^ r3), 0x1414); S22 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c2 ^ r0), (c3 ^ r1), 0x3636); tmp = __byte_perm((c0 ^ r2), (c1 ^ r3), 0x1414); S23 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c3 ^ r0), (c0 ^ r1), 0x3636);
		tmp = __byte_perm((c1 ^ r2), (c2 ^ r3), 0x1414);
		S24 = __byte_perm(tmp2, tmp, 0x3254);

		CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02);
		SMIX(S18, S19, S20, S21);
		CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35);
		SMIX(S15, S16, S17, S18);
		CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32);
		SMIX(S12, S13, S14, S15);

		S34 ^= S12;
		S12 = (64 << 3);
		S20 ^= S12; S13 ^= S00; S16 ^= S03; S19 ^= S06;
		S09 ^= S13; S10 ^= S14; S11 ^= S15; S27 ^= S13; S28 ^= S14; S29 ^= S15;
		tmp = (*(mixtabs + ((__byte_perm(S09, 0, 0x4443)))));  c0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S09, 0, 0x4442))))); c0 ^= tmp; r1 = tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S09, 0, 0x4441))))); c0 ^= tmp; r2 = tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S09, 0, 0x4440))))); c0 ^= tmp; r3 = tmp; tmp = (*(mixtabs + ((__byte_perm(S10, 0, 0x4443))))); c1 = tmp; r0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S10, 0, 0x4442))))); c1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S10, 0, 0x4441))))); c1 ^= tmp; r2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S10, 0, 0x4440))))); c1 ^= tmp; r3 ^= tmp; tmp = (*(mixtabs + ((__byte_perm(S11, 0, 0x4443))))); c2 = tmp; r0 ^= tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S11, 0, 0x4442))))); c2 ^= tmp; r1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S11, 0, 0x4441))))); c2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S11, 0, 0x4440))))); c2 ^= tmp; r3 ^= tmp;
		r0 ^= 0x63633297UL;
		r1 ^= 0x97636332UL;
		r2 ^= 0x5ec77777UL;
		c3 = (0x63633297UL ^ 0x97636332UL ^ 0x5ec77777UL ^ 0x63329763);
		tmp2 = __byte_perm((c0 ^ r0), (c1 ^ r1), 0x3636); tmp = __byte_perm((c2 ^ r2), (c3 ^ r3), 0x1414); S09 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c1 ^ r0), (c2 ^ r1), 0x3636); tmp = __byte_perm((c3 ^ r2), (c0 ^ r3), 0x1414); S10 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c2 ^ r0), (c3 ^ r1), 0x3636); tmp = __byte_perm((c0 ^ r2), (c1 ^ r3), 0x1414); S11 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c3 ^ r0), (c0 ^ r1), 0x3636); tmp = __byte_perm((c1 ^ r2), (c2 ^ r3), 0x1414); S12 = __byte_perm(tmp2, tmp, 0x3254);

		CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26);
		SMIX(S06, S07, S08, S09);
		CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23);
		SMIX(S03, S04, S05, S06);
		CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
		SMIX(S00, S01, S02, S03);

		//#pragma unroll
		for (int i = 0; i < 32; i++) {
			ROR3;
			CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
			SMIX(S00, S01, S02, S03);
		}

		#pragma	unroll
		for (int i = 0; i < 13; i++)
		{
			S04 ^= S00;
			S09 ^= S00;
			S18 ^= S00;
			S27 ^= S00;
			SMIX(S27, S28, S29, S30);
			S31 ^= S27;
			S01 ^= S27;
			S09 ^= S27;
			S18 ^= S27;
			SMIX(S18, S19, S20, S21);
			S22 ^= S18;
			S28 ^= S18;
			S01 ^= S18;
			S09 ^= S18;
			SMIX(S09, S10, S11, S12);
			S13 ^= S09;
			S19 ^= S09;
			S28 ^= S09;
			S01 ^= S09;
			SMIX(S01, S02, S03, S04);
			ROL1;
		}
		S04 ^= S00;
		S09 ^= S00;
		S18 ^= S00;
		S27 ^= S00;

		Hash[0] = cuda_swab32(S01);
		Hash[1] = cuda_swab32(S02);
		Hash[2] = cuda_swab32(S03);
		Hash[3] = cuda_swab32(S04);
		Hash[4] = cuda_swab32(S09);
		Hash[5] = cuda_swab32(S10);
		Hash[6] = cuda_swab32(S11);
		Hash[7] = cuda_swab32(S12);
		Hash[8] = cuda_swab32(S18);
		Hash[9] = cuda_swab32(S19);
		Hash[10] = cuda_swab32(S20);
		Hash[11] = cuda_swab32(S21);
		Hash[12] = cuda_swab32(S27);
		Hash[13] = cuda_swab32(S28);
		Hash[14] = cuda_swab32(S29);
		Hash[15] = cuda_swab32(S30);
	}
}

//__launch_bounds__(256, 3)
__global__
void x13_fugue512_gpu_hash_64_final(const uint32_t threads, const uint32_t startNounce, const uint32_t *const __restrict__ g_hash, uint32_t *const __restrict__ d_nonce)
{
	__shared__ uint32_t mixtabs[1024];

	const uint32_t precalc[38] = { 0x8807a57eUL, 0xe616af75UL, 0xc5d3e4dbUL, 0xac9ab027UL,
		0xd915f117UL, 0xb6eecc54UL, 0x06e8020bUL, 0x4a92efd1UL,
		0xaac6e2c9UL, 0xddb21398UL, 0xcae65838UL, 0x437f203fUL,
		0x25ea78e7UL, 0x4c0a2cc1UL, 0xda6ed11dUL, 0xe13e3567UL,
		0xd915f117UL, 0x4a92efd1UL, 0xcae65838UL, 0xd915f117UL,
		0x9ae23283UL, 0x0361b92dUL, 0x4c92d8edUL, 0xafaf608aUL,
		0x79d5d51dUL, 0xf6274f4fUL, 0x59947f59UL, 0x947f5959UL,
		0x7f595994UL, 0x5959947fUL, 0x63633297UL, 0x97636332UL,
		0x32976363UL, (0x63633297UL ^ 0x97636332UL ^ 0x32976363UL ^ 0x63329763UL),
		0x63633297UL, 0x97636332UL, 0x5ec77777UL, (0x63633297UL ^ 0x97636332UL ^ 0x5ec77777UL ^ 0x63329763)
	};

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread< threads)
	{
		if (threadIdx.x < 256)
		{
			mixtabs[threadIdx.x] = mixTab0Tex[threadIdx.x];
			mixtabs[256 + threadIdx.x] = ROL24(mixtabs[threadIdx.x]);
			mixtabs[(512 + threadIdx.x)] = ROL16(mixtabs[threadIdx.x]);
			mixtabs[(768 + threadIdx.x)] = ROL8(mixtabs[threadIdx.x]);
		}
		uint32_t backup = pTarget[7];
		const uint32_t nounce =  (startNounce + thread);
		const int hashPosition = nounce - startNounce;
		const uint32_t *h = &g_hash[hashPosition * 16];
		uint32_t Hash[16];
#pragma unroll 16
		for (int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(h[i]);

		uint32_t S00, S01, S02, S03, S04, S05, S06, S07, S08, S09;
		uint32_t S10, S11, S12, S13, S14, S15, S16, S17, S18, S19;
		uint32_t S20, S21, S22, S23, S24, S25, S26, S27, S28, S29;
		uint32_t S30, S31, S32, S33, S34, S35;

		uint32_t B33, B34, B35;


		S02 = S03 = S05 = S06 = S09 = S10 = S11 = S12 = S13 = S14 = S16 = S17 = S18 = S19 = 0;
		S20 = precalc[0]; S21 = precalc[1]; S22 = precalc[2]; S23 = precalc[3];
		S24 = precalc[4]; S25 = precalc[5]; S26 = precalc[6]; S27 = precalc[7];
		S28 = precalc[8]; S29 = precalc[9]; S30 = precalc[10]; S31 = precalc[11];
		S32 = precalc[12]; S33 = precalc[13]; S34 = precalc[14]; S35 = precalc[15];

		S01 = precalc[16];
		S04 = precalc[17];
		S07 = precalc[18];
		S15 = precalc[19];
		S00 = Hash[0];
		S08 = Hash[0];

		uint32_t c0 = precalc[20];
		uint32_t c1 = precalc[21];
		uint32_t c2 = precalc[22];
		uint32_t r0, r1, r2;
		uint32_t tmp, tmp2, c3;

		tmp = mixtabs[__byte_perm(S00, 0, 17475)]; c3 = tmp; r0 = precalc[23] ^ tmp;
		tmp = mixtabs[256 + __byte_perm(S00, 0, 17474)]; c3 ^= tmp; r1 = precalc[24] ^ tmp;
		tmp = mixtabs[512 + __byte_perm(S00, 0, 17473)]; c3 ^= tmp; r2 = precalc[25]  ^ tmp;
		tmp = mixtabs[768 + __byte_perm(S00, 0, 17472)]; c3 ^= tmp;
		tmp2 = __byte_perm(c0 ^ r0, c1 ^ r1, 13878);
		tmp = __byte_perm(c2 ^ r2, c3 ^ precalc[26] , 5140);
		S33 = __byte_perm(tmp2, tmp, 12884);
		r0 = ROL8(r0); r1 = ROL8(r1);
		r2 = ROL8(r2);
		tmp2 = __byte_perm(c1 ^ r0, c2 ^ r1, 13878);
		tmp = __byte_perm(c3 ^ r2, c0 ^ precalc[27] , 5140);
		S34 = __byte_perm(tmp2, tmp, 12884);
		r0 = ROL8(r0); r1 = ROL8(r1);
		r2 = ROL8(r2);
		tmp2 = __byte_perm(c2 ^ r0, c3 ^ r1, 13878);
		tmp = __byte_perm(c0 ^ r2, c1 ^ precalc[28] , 5140);
		S35 = __byte_perm(tmp2, tmp, 12884);
		r0 = ROL8(r0); r1 = ROL8(r1);
		r2 = ROL8(r2);
		tmp2 = __byte_perm(c3 ^ r0, c0 ^ r1, 13878);
		tmp = __byte_perm(c1 ^ r2, c2 ^ precalc[29] , 5140);
		S00 = __byte_perm(tmp2, tmp, 12884);

		CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14);
		SMIX(S30, S31, S32, S33);
		CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11);
		SMIX(S27, S28, S29, S30);
		CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08);
		SMIX(S24, S25, S26, S27);

		TIX4(Hash[1], S24, S25, S28, S31, S32, S10, S12, S15, S18);
		CMIX36(S21, S22, S23, S25, S26, S27, S03, S04, S05);
		SMIX(S21, S22, S23, S24);
		CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02);
		SMIX(S18, S19, S20, S21); 
		CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35);
		SMIX(S15, S16, S17, S18);
		CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32);
		SMIX(S12, S13, S14, S15);
			
		TIX4(Hash[2], S12, S13, S16, S19, S20, S34, S00, S03, S06); 
		CMIX36(S09, S10, S11, S13, S14, S15, S27, S28, S29); 
		SMIX(S09, S10, S11, S12); 
		CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26); 
		SMIX(S06, S07, S08, S09); 
		CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23); 
		SMIX(S03, S04, S05, S06); 
		CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20); 
		SMIX(S00, S01, S02, S03); 

#pragma unroll
		for (int i = 3; i < (5 * 3); i += 3)
		{
			FUGUE512_3((Hash[i]), (Hash[i + 1]), (Hash[i + 2]));
		}
		TIX4(Hash[0xF], S00, S01, S04, S07, S08, S22, S24, S27, S30);
		CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17);
		SMIX(S33, S34, S35, S00);
		CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14);
		SMIX(S30, S31, S32, S33);
		CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11);
		SMIX(S27, S28, S29, S30);
		CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08);
		SMIX(S24, S25, S26, S27);

		S10 ^= S24;
		S25 ^= S12; S28 ^= S15; S31 ^= S18;
		S21 ^= S25; S22 ^= S26; S23 ^= S27; S03 ^= S25; S04 ^= S26; S05 ^= S27;
		tmp = (*(mixtabs + ((__byte_perm(S21, 0, 0x4443))))); c0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S21, 0, 0x4442))))); c0 ^= tmp; r1 = tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S21, 0, 0x4441))))); c0 ^= tmp; r2 = tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S21, 0, 0x4440))))); c0 ^= tmp; uint32_t r3 = tmp; tmp = (*(mixtabs + ((__byte_perm(S22, 0, 0x4443))))); c1 = tmp; r0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S22, 0, 0x4442))))); c1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S22, 0, 0x4441))))); c1 ^= tmp; r2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S22, 0, 0x4440))))); c1 ^= tmp; r3 ^= tmp; tmp = (*(mixtabs + ((__byte_perm(S23, 0, 0x4443))))); c2 = tmp; r0 ^= tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S23, 0, 0x4442))))); c2 ^= tmp; r1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S23, 0, 0x4441))))); c2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S23, 0, 0x4440))))); c2 ^= tmp; r3 ^= tmp;
		r0 ^= precalc[30];
		r1 ^= precalc[31];
		r2 ^= precalc[32];
		c3 = precalc[33];
		tmp2 = __byte_perm((c0 ^ r0), (c1 ^ r1), 0x3636); tmp = __byte_perm((c2 ^ r2), (c3 ^ r3), 0x1414); S21 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c1 ^ r0), (c2 ^ r1), 0x3636); tmp = __byte_perm((c3 ^ r2), (c0 ^ r3), 0x1414); S22 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c2 ^ r0), (c3 ^ r1), 0x3636); tmp = __byte_perm((c0 ^ r2), (c1 ^ r3), 0x1414); S23 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c3 ^ r0), (c0 ^ r1), 0x3636);
		tmp = __byte_perm((c1 ^ r2), (c2 ^ r3), 0x1414);
		S24 = __byte_perm(tmp2, tmp, 0x3254);

		CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02);
		SMIX(S18, S19, S20, S21);
		CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35);
		SMIX(S15, S16, S17, S18);
		CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32);
		SMIX(S12, S13, S14, S15);

        S34 ^= S12; 
		S12 = (64 << 3); 
		S20 ^= S12; S13 ^= S00; S16 ^= S03; S19 ^= S06; 
		S09 ^= S13; S10 ^= S14; S11 ^= S15; S27 ^= S13; S28 ^= S14; S29 ^= S15;
		tmp = (*(mixtabs + ((__byte_perm(S09, 0, 0x4443)))));  c0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S09, 0, 0x4442))))); c0 ^= tmp; r1 = tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S09, 0, 0x4441))))); c0 ^= tmp; r2 = tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S09, 0, 0x4440))))); c0 ^= tmp; r3 = tmp; tmp = (*(mixtabs + ((__byte_perm(S10, 0, 0x4443))))); c1 = tmp; r0 = tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S10, 0, 0x4442))))); c1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S10, 0, 0x4441))))); c1 ^= tmp; r2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S10, 0, 0x4440))))); c1 ^= tmp; r3 ^= tmp; tmp = (*(mixtabs + ((__byte_perm(S11, 0, 0x4443))))); c2 = tmp; r0 ^= tmp; tmp = (*(mixtabs + (256 + (__byte_perm(S11, 0, 0x4442))))); c2 ^= tmp; r1 ^= tmp; tmp = (*(mixtabs + (512 + (__byte_perm(S11, 0, 0x4441))))); c2 ^= tmp; tmp = (*(mixtabs + (768 + (__byte_perm(S11, 0, 0x4440))))); c2 ^= tmp; r3 ^= tmp; 
		r0 ^= precalc[34];
		r1 ^= precalc[35];
		r2 ^= precalc[36];
		c3 = precalc[37];
		tmp2 = __byte_perm((c0 ^ r0), (c1 ^ r1), 0x3636); tmp = __byte_perm((c2 ^ r2), (c3 ^ r3), 0x1414); S09 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c1 ^ r0), (c2 ^ r1), 0x3636); tmp = __byte_perm((c3 ^ r2), (c0 ^ r3), 0x1414); S10 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c2 ^ r0), (c3 ^ r1), 0x3636); tmp = __byte_perm((c0 ^ r2), (c1 ^ r3), 0x1414); S11 = __byte_perm(tmp2, tmp, 0x3254); r0 = __funnelshift_l((r0), (r0), (8)); r1 = __funnelshift_l((r1), (r1), (8)); r2 = __funnelshift_l((r2), (r2), (8)); r3 = __funnelshift_l((r3), (r3), (8)); tmp2 = __byte_perm((c3 ^ r0), (c0 ^ r1), 0x3636); tmp = __byte_perm((c1 ^ r2), (c2 ^ r3), 0x1414); S12 = __byte_perm(tmp2, tmp, 0x3254);

		CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26);
		SMIX(S06, S07, S08, S09);
		CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23);
		SMIX(S03, S04, S05, S06);
		CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
		SMIX(S00, S01, S02, S03);

		for (int i = 0; i < 32; i++) {
			ROR3;
			CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
			SMIX(S00, S01, S02, S03);
		}
		#pragma unroll
		for (int i = 0; i < 11; i++) 
		{
			S04 ^= S00;
			S09 ^= S00;
			S18 ^= S00;
			S27 ^= S00;
			SMIX(S27, S28, S29, S30);
			S31 ^= S27; 
			S01 ^= S27;
			S09 ^= S27; 
			S18 ^= S27;
			SMIX(S18, S19, S20, S21);
			S22 ^= S18;
			S28 ^= S18;
			S01 ^= S18;
			S09 ^= S18;
			SMIX(S09, S10, S11, S12);
			S13 ^= S09;
			S19 ^= S09;
			S28 ^= S09;
			S01 ^= S09; 
			SMIX(S01, S02, S03, S04);
			ROL1;
		}
		S04 ^= S00;
		S09 ^= S00;
		S18 ^= S00;
		S27 ^= S00;
		SMIX(S27, S28, S29, S30);
		S31 ^= S27;
		S01 ^= S27;
		S09 ^= S27;
		S18 ^= S27;
		SMIX(S18, S19, S20, S21);
		S22 ^= S18;
		S28 ^= S18;
		S01 ^= S18;
		S09 ^= S18;
		SMIX(S09, S10, S11, S12);
		S13 ^= S09;
		S19 ^= S09;
		S28 ^= S09;
		S01 ^= S09;
		SMIX0(S01, S02, S03, S04);
		S10 ^= S01;
		S19 ^= S01;
		S28 ^= S01;
		SMIX0(S28, S29, S30, S31);
		S10 ^= S28;
		S19 ^= S28;
		SMIX0(S19, S20, S21, S22);
		S10 ^= S19;
		SMIX0(S10, S11, S12, S13);
		S14 ^= S10;
		if (cuda_swab32(S14) <= backup)
		{
			uint32_t tmp = atomicExch(d_nonce, nounce);
			if (tmp != 0xffffffff)
				d_nonce[1] = tmp;
		}
	}
}

__host__ void x13_fugue512_cpu_init(int thr_id, uint32_t threads)
{
	cudaMalloc(&d_nonce[thr_id], 2*sizeof(uint32_t));
}
__host__ void x13_fugue512_cpu_setTarget(const void *ptarget)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, ptarget, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

}

__host__ void  x13_fugue512_cpu_free(int32_t thr_id)
{
	cudaFree(pTarget);
	cudaFreeHost(&d_nonce[thr_id]);
}

__host__ void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 128;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// fprintf(stderr, "threads=%d, %d blocks, %d threads per block, %d bytes shared\n", threads, grid.x, block.x, shared_size);

	x13_fugue512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash);
}
__host__ void x13_fugue512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *res)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	cudaMemset(d_nonce[thr_id], 0xff, 2*sizeof(uint32_t));

	x13_fugue512_gpu_hash_64_final << <grid, block>> >(threads, startNounce, d_hash, d_nonce[thr_id]);
	cudaMemcpy(res, d_nonce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
