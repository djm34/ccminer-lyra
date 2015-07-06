
/*
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include <stdint.h>
#include <memory.h>
*/
#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h" 

extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

// __device__  uint4 *  S;
 __constant__  uint32 *prevstate;
 __device__ uint32 *B; 
 __device__ ulonglong8to16 *state2;
 __device__ uint8 *sha256test;
uint32_t *d_YNonce[MAX_GPUS];
__constant__  uint32_t pTarget[8];
__constant__  uint32_t  c_data[32];
__constant__  uint16 shapad;




///////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// sha256 Transform function /////////////////////////


static __constant__ uint16 pad1 = 
{
	0x36363636, 0x36363636, 0x36363636, 0x36363636,
	0x36363636, 0x36363636, 0x36363636, 0x36363636,
	0x36363636, 0x36363636, 0x36363636, 0x36363636,
	0x36363636, 0x36363636, 0x36363636, 0x36363636
};

static __constant__ uint16 pad2 = 
{
	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,
	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,
	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c,
	0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c, 0x5c5c5c5c
};

static __constant__ uint16 pad5 =
{
	0x00000001, 0x80000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00002220
};



static __constant__ uint16 padsha80 =
{
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000280
};

static __constant__ uint8 pad4 =
{
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000300
};



static __constant__  uint8 H256 = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372,
	0xA54FF53A, 0x510E527F, 0x9B05688C,
	0x1F83D9AB, 0x5BE0CD19
};

static  __constant__  uint32_t Ksha[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
	0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
	0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
	0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
	0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
	0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
	0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
	0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
	0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

static __device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	uint32_t r1 = ROTR32(x, 2);
	uint32_t r2 = ROTR32(x, 13);
	uint32_t r3 = ROTR32(x, 22);
	return xor3b(r1, r2, r3);
}


static __device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	uint32_t r1 = ROTR32(x, 6);
	uint32_t r2 = ROTR32(x, 11);
	uint32_t r3 = ROTR32(x, 25);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	uint64_t r1 = ROTR32(x, 7);
	uint64_t r2 = ROTR32(x, 18);
	uint64_t r3 = shr_t32(x, 3);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	uint64_t r1 = ROTR32(x, 17);
	uint64_t r2 = ROTR32(x, 19);
	uint64_t r3 = shr_t32(x, 10);
	return xor3b(r1, r2, r3);
}

static __device__ __forceinline__ void sha2_step1(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d, const uint32_t e,
	const uint32_t f, const uint32_t g, uint32_t &h, const uint32_t in, const uint32_t Kshared)
{
	uint32_t t1, t2;
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a, b, c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

static __device__ __forceinline__ void sha2_step2(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d, const uint32_t e,
	const uint32_t f, const uint32_t g, uint32_t &h, uint32_t* in, const uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1, t2;

	int pcidx1 = (pc - 2) & 0xF;
	int pcidx2 = (pc - 7) & 0xF;
	int pcidx3 = (pc - 15) & 0xF;
	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ssg2_1(inx1);
	uint32_t ssg20 = ssg2_0(inx3);
	uint32_t vxandx = xandx(e, f, g);
	uint32_t bsg21 = bsg2_1(e);
	uint32_t bsg20 = bsg2_0(a);
	uint32_t andorv = andor32(a, b, c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}



#define SALSA(a,b,c,d) { \
    t =a+d; b^=rotate(t,  7);    \
    t =b+a; c^=rotate(t,  9);    \
    t =c+b; d^=rotate(t, 13);    \
    t =d+c; a^=rotate(t, 18);     \
}

#define SALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s5,state.s9,state.sd,state.s1); \
SALSA(state.sa,state.se,state.s2,state.s6); \
SALSA(state.sf,state.s3,state.s7,state.sb); \
SALSA(state.s0,state.s1,state.s2,state.s3); \
SALSA(state.s5,state.s6,state.s7,state.s4); \
SALSA(state.sa,state.sb,state.s8,state.s9); \
SALSA(state.sf,state.sc,state.sd,state.se); \
} 

#define uSALSA_CORE(state) { \
\
SALSA(state.s0,state.s4,state.s8,state.sc); \
SALSA(state.s1,state.s5,state.s9,state.sd); \
SALSA(state.s2,state.s6,state.sa,state.se); \
SALSA(state.s3,state.s7,state.sb,state.sf); \
SALSA(state.s0,state.sd,state.sa,state.s7); \
SALSA(state.s1,state.se,state.sb,state.s4); \
SALSA(state.s2,state.sf,state.s8,state.s5); \
SALSA(state.s3,state.sc,state.s9,state.s6); \
} 


#define shuffle(stat,state) { \
stat.s0 = state.s0; \
stat.s1 = state.s5; \
stat.s2 = state.sa; \
stat.s3 = state.sf; \
stat.s4 = state.s4; \
stat.s5 = state.s9; \
stat.s6 = state.se; \
stat.s7 = state.s3; \
stat.s8 = state.s8; \
stat.s9 = state.sd; \
stat.sa = state.s2; \
stat.sb = state.s7; \
stat.sc = state.sc; \
stat.sd = state.s1; \
stat.se = state.s6; \
stat.sf = state.sb; \
}
#define unshuffle(state,X) { \
    state.s0 = X.s0; \
    state.s1 = X.sd; \
    state.s2 = X.sa; \
    state.s3 = X.s7; \
    state.s4 = X.s4; \
    state.s5 = X.s1; \
    state.s6 = X.se; \
    state.s7 = X.sb; \
    state.s8 = X.s8; \
    state.s9 = X.s5; \
    state.sa = X.s2; \
    state.sb = X.sf; \
    state.sc = X.sc; \
    state.sd = X.s9; \
    state.se = X.s6; \
    state.sf = X.s3; \
}

static __device__ __forceinline__
void sha256_Transform(uint16 in[1], uint8 &r) // also known as sha2_round_body
{
	uint32_t a = r.s0;
	uint32_t b = r.s1;
	uint32_t c = r.s2;
	uint32_t d = r.s3;
	uint32_t e = r.s4;
	uint32_t f = r.s5;
	uint32_t g = r.s6;
	uint32_t h = r.s7;

	sha2_step1(a, b, c, d, e, f, g, h, in[0].s0, Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s1, Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].s2, Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].s3, Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].s4, Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].s5, Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].s6, Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].s7, Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[0].s8, Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s9, Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].sa, Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].sb, Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].sc, Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].sd, Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].se, Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].sf, Ksha[15]);

#pragma unroll 3
	for (int i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 15, Ksha[31 + 16 * i]);

	}

	r.s0 += a;
	r.s1 += b;
	r.s2 += c;
	r.s3 += d;
	r.s4 += e;
	r.s5 += f;
	r.s6 += g;
	r.s7 += h;
}

static __device__ __forceinline__
uint8 sha256_Transform2(uint16 in[1], const uint8 &r) // also known as sha2_round_body
{
	uint8 tmp = r;
#define a  tmp.s0
#define b  tmp.s1
#define c  tmp.s2
#define d  tmp.s3
#define e  tmp.s4
#define f  tmp.s5
#define g  tmp.s6
#define h  tmp.s7

	sha2_step1(a, b, c, d, e, f, g, h, in[0].s0, Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s1, Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].s2, Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].s3, Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].s4, Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].s5, Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].s6, Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].s7, Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[0].s8, Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s9, Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].sa, Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].sb, Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].sc, Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].sd, Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].se, Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].sf, Ksha[15]);

#pragma unroll 3
	for (int i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 15, Ksha[31 + 16 * i]);

	}
#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
	return (r + tmp);
}


static __device__ __forceinline__
uint8 sha256_Transform3(uint32_t nonce,uint32_t next, const uint8 &r) // also known as sha2_round_body
{
	uint8 tmp = r;
	uint16 in[1]={shapad};
	in[0].s3=nonce;
	in[0].s4=next;
#define a  tmp.s0
#define b  tmp.s1
#define c  tmp.s2
#define d  tmp.s3
#define e  tmp.s4
#define f  tmp.s5
#define g  tmp.s6
#define h  tmp.s7

	sha2_step1(a, b, c, d, e, f, g, h, in[0].s0, Ksha[0]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s1, Ksha[1]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].s2, Ksha[2]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].s3, Ksha[3]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].s4, Ksha[4]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].s5, Ksha[5]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].s6, Ksha[6]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].s7, Ksha[7]);
	sha2_step1(a, b, c, d, e, f, g, h, in[0].s8, Ksha[8]);
	sha2_step1(h, a, b, c, d, e, f, g, in[0].s9, Ksha[9]);
	sha2_step1(g, h, a, b, c, d, e, f, in[0].sa, Ksha[10]);
	sha2_step1(f, g, h, a, b, c, d, e, in[0].sb, Ksha[11]);
	sha2_step1(e, f, g, h, a, b, c, d, in[0].sc, Ksha[12]);
	sha2_step1(d, e, f, g, h, a, b, c, in[0].sd, Ksha[13]);
	sha2_step1(c, d, e, f, g, h, a, b, in[0].se, Ksha[14]);
	sha2_step1(b, c, d, e, f, g, h, a, in[0].sf, Ksha[15]);

#pragma unroll 3
	for (int i = 0; i<3; i++) {

		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 0, Ksha[16 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 1, Ksha[17 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 2, Ksha[18 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 3, Ksha[19 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 4, Ksha[20 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 5, Ksha[21 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 6, Ksha[22 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 7, Ksha[23 + 16 * i]);
		sha2_step2(a, b, c, d, e, f, g, h, (uint32_t*)in, 8, Ksha[24 + 16 * i]);
		sha2_step2(h, a, b, c, d, e, f, g, (uint32_t*)in, 9, Ksha[25 + 16 * i]);
		sha2_step2(g, h, a, b, c, d, e, f, (uint32_t*)in, 10, Ksha[26 + 16 * i]);
		sha2_step2(f, g, h, a, b, c, d, e, (uint32_t*)in, 11, Ksha[27 + 16 * i]);
		sha2_step2(e, f, g, h, a, b, c, d, (uint32_t*)in, 12, Ksha[28 + 16 * i]);
		sha2_step2(d, e, f, g, h, a, b, c, (uint32_t*)in, 13, Ksha[29 + 16 * i]);
		sha2_step2(c, d, e, f, g, h, a, b, (uint32_t*)in, 14, Ksha[30 + 16 * i]);
		sha2_step2(b, c, d, e, f, g, h, a, (uint32_t*)in, 15, Ksha[31 + 16 * i]);

	}
#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
	return (r + tmp);
}

//////////////////////////////// end sha transform mechanism ////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ uint8 sha256_80(uint32_t nonce)
{
	//	uint32_t in[16], buf[8];
	uint16 in[1] = { 0 };
	uint8 buf;

	in[0] = ((uint16*)c_data)[0];
	buf = H256;

	sha256_Transform(in, buf);
	in[0] = padsha80;
	in[0].s0 = c_data[16];
	in[0].s1 = c_data[17];
	in[0].s2 = c_data[18];
	in[0].s3 = nonce;
	

	sha256_Transform(in, buf);
	return buf;
}



static __forceinline__ __device__ uint16 salsa20_8(const uint16 &X)
{
	uint16 state=X;
	uint32_t t; 
	for (int i = 0; i < 4; ++i) { uSALSA_CORE(state); }
	return(X + state);
}

static __forceinline__ __device__ void block_pwxform_long(int thread, ulonglong2to8 &Bout)
{

		ulonglong2 vec = Bout.l0;

		for (int i = 0; i < 6; i++)
		{
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout.l0 = vec;
        vec = Bout.l1;
		for (int i = 0; i < 6; i++)
		{
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout.l1 = vec;

		vec = Bout.l2;
		for (int i = 0; i < 6; i++)
		{
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout.l2 = vec;
		vec = Bout.l3;
		for (int i = 0; i < 6; i++)
		{
			ulonglong2 p0, p1;
			uint2 x = vectorize((vec.x >> 4) & 0x000000FF000000FF);
			p0 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread))[x.x]);
			madd4long2(vec, p0);
			p1 = __ldg2(&((ulonglong2*)(prevstate + 64 * thread + 32))[x.y]);


			vec ^= p1;
		}
		Bout.l3 = vec;


}


static __forceinline__ __device__ void blockmix_salsa8_small2(uint32 &Bin)
{
	uint16 X = Bin.hi;
	X ^= Bin.lo;
	X = salsa20_8(X);
	Bin.lo = X;
	X ^= Bin.hi;
	X = salsa20_8(X);
	Bin.hi = X;
}



static __forceinline__ __device__ void blockmix_pwxform3(int thread, ulonglong2to8 *Bin)
{
	Bin[0] ^= Bin[15];
	block_pwxform_long(thread, Bin[0]);

	for (int i = 1; i < 16; i++)
	{
		Bin[i] ^= Bin[i - 1];
		block_pwxform_long(thread, Bin[i]);
	}
	((uint16*)Bin)[15] = salsa20_8(((uint16*)Bin)[15]);
//	Bin[15] = salsa20_8_long(Bin[15]);
}


__global__ __launch_bounds__(16, 1) void yescrypt_gpu_hash_k0(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);
	
    
//	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint16 in[1];
		uint8 state1, state2;
		uint8 passwd = sha256_80(nonce);
		in[0].lo = pad1.lo ^ passwd;
		in[0].hi = pad1.hi;
		state1 = sha256_Transform2(in, H256);
		in[0].lo = pad2.lo ^ passwd;
		in[0].hi = pad2.hi;
		state2 = sha256_Transform2(in, H256);
		in[0] = ((uint16*)c_data)[0];
		///HMAC_SHA256_update(salt)
		state1 = sha256_Transform2(in, state1);
#pragma unroll	
		for (int i = 0; i<8; i++)
		{
			uint32 result;

			in[0].lo = sha256_Transform3(nonce,4*i+1, state1);
			in[0].hi = pad4;
			result.lo.lo = swapvec(sha256_Transform2(in, state2));
			if (i == 0) (sha256test + thread)[0] = result.lo.lo;

			in[0].lo = sha256_Transform3(nonce,4*i+2, state1);
			in[0].hi = pad4;
			result.lo.hi = swapvec(sha256_Transform2(in, state2));


			in[0].lo = sha256_Transform3(nonce,4*i+3, state1);
			in[0].hi = pad4;
			result.hi.lo = swapvec(sha256_Transform2(in, state2));

			in[0].lo = sha256_Transform3(nonce,4*i+4, state1);
			in[0].hi = pad4;
			result.hi.hi = swapvec(sha256_Transform2(in, state2));


			shuffle((B + 8 * thread)[i].lo, result.lo);
			shuffle((B + 8 * thread)[i].hi, result.hi);
		}

 
	}
}

__global__ __launch_bounds__(16, 1) void yescrypt_gpu_hash_k1(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);


//	if (thread < threads)
	{

//		smix1_first(thread);
		uint32 X;

#define Bdev(x) (B+8*thread)[x]
#define state(x) (prevstate+64*thread)[x]
		X = Bdev(0);
		state(0) = X; 
		blockmix_salsa8_small2(X);
		state(1) = X;
		blockmix_salsa8_small2(X);



		uint32_t n = 1;
		for (uint32_t i = 2; i < 64; i ++)
		{

			state(i) = X;

			if ((i&(i - 1)) == 0) n = n << 1;

			uint32_t j = X.hi.s0 & (n - 1);

			j += i - n;
			X ^= __ldg32b(&state(j));

			blockmix_salsa8_small2(X);
		}

		Bdev(0) = X;
#undef Bdev
#undef state

	}
}
#define thelength  8
#define vectype ulonglong8to16

__global__ __launch_bounds__(16, 1) void yescrypt_gpu_hash_k2c(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);


//	if (thread < threads)
	{

		//		smix1_second(thread);
		vectype X[thelength]; //,Z;
       uint32_t length = thelength;
		uint32_t shift = length * 2048 * thread;
 
#define Bdev(x) (B+8*thread)[x]

	
		for (int i = 0; i<8; i++) {
			((uint32*)X)[i] = __ldg32b(&Bdev(i));
		}

		for (int i = 0; i<length; i++)
				(state2+shift)[i] = X[i];

		blockmix_pwxform3(thread, (ulonglong2to8*)X);

		for (int i = 0; i<length; i++)
			(state2 + shift+length)[i] = X[i];

		blockmix_pwxform3(thread, (ulonglong2to8*)X); 
		int n = 1;

		for (int i = 2; i < 2048; i ++)
		{


			for (int k = 0; k<length; k++)
			(state2 + shift + length * i)[k] = X[k];


			if ((i&(i - 1)) == 0) n = n << 1;

			uint32_t j = ((uint32*)X)[7].hi.s0 & (n - 1);
			j += i - n;

			for (int k = 0; k < 64; k++) 	
			((ulonglong2*)X)[k] ^= __ldg2(&((ulonglong2*)(state2 + shift + length * j))[k]);
						

			blockmix_pwxform3(thread, (ulonglong2to8*)X);

		}
		for (int i = 0; i<8; i++) {
			(B + 8 * thread)[i] = ((uint32*)X)[i];
		}





		/////////////////////////////////////////////////
	}
}

__global__ __launch_bounds__(16, 1) void yescrypt_gpu_hash_k2c1(int threads, uint32_t startNonce)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);


	if (thread < threads)
	{


		vectype X[thelength]; //,Z;
		uint32_t length = thelength;
		uint32_t shift = length * 2048 * thread;

#define Bdev(x) (B+8*thread)[x]
#define BigStore(s,i) (state2 + shift + s)[i]


		for (int i = 0; i<8; i++) {
			((uint32*)X)[i] = __ldg32b(&Bdev(i));
		}
		
		//#pragma unroll
		for (int z = 0; z < 684; z++)
		{
		
			uint32_t j = ((uint32*)X)[7].hi.s0 & 2047;

			for (int k = 0; k < 64; k++)
				((ulonglong2*)X)[k] ^= __ldg2(&((ulonglong2*)(state2 + shift + length * j))[k]);


		if(z<682)
	for (int k = 0; k<length; k++)
				BigStore(length * j, k) = X[k];

			blockmix_pwxform3(thread, (ulonglong2to8*)X);

		}


		for (int i = 0; i<8; i++) {
//			((uint32*)X)[i] = Bdev(i);
			unshuffle(Bdev(i).lo, ((uint32*)X)[i].lo);
			unshuffle(Bdev(i).hi, ((uint32*)X)[i].hi);
		}

	}
}

__global__ __launch_bounds__(16, 1) void yescrypt_gpu_hash_k5(int threads, uint32_t startNonce, uint32_t *nonceVector)
{

	int thread = (blockDim.x * blockIdx.x + threadIdx.x);


		const uint32_t nonce = startNonce + thread;



		uint16 in[1];
		uint8 state1, state2;

		uint8 swpass = (sha256test + thread)[0];
#define Bdev(x) (B+8*thread)[x]
		swpass = swapvec(swpass);
		in[0].lo = pad1.lo ^ swpass;
		in[0].hi = pad1.hi;

		state1 = sha256_Transform2(in, H256);

		in[0].lo = pad2.lo ^ swpass;
		in[0].hi = pad2.hi;
		state2 = sha256_Transform2(in, H256);

		for (int i = 0; i<8; i++) {
			in[0] = __ldg16b(&Bdev(i).lo);
			in[0] = swapvec(in[0]);
			state1 = sha256_Transform2(in, state1);
			in[0] = __ldg16b(&Bdev(i).hi);
			in[0] = swapvec(in[0]);
			state1 = sha256_Transform2(in, state1);
		}
		in[0] = pad5;
		state1 = sha256_Transform2(in, state1);
		in[0].lo = state1;
		in[0].hi = pad4;
		uint8 res = sha256_Transform2(in, state2);

		//hmac and final sha
		state1 = state2 = H256;
		in[0].lo = pad1.lo ^ res;
		in[0].hi = pad1.hi;
		state1 = sha256_Transform2(in, state1);
		in[0].lo = pad2.lo ^ res;
		in[0].hi = pad2.hi;
		state2 = sha256_Transform2(in, state2);
		in[0] = ((uint16*)c_data)[0];
		state1 = sha256_Transform2(in, state1);
		in[0] = padsha80;
		in[0].s0 = c_data[16];
		in[0].s1 = c_data[17];
		in[0].s2 = c_data[18];
		in[0].s3 = nonce;
		in[0].sf = 0x480;
		state1 = sha256_Transform2(in, state1);
		in[0].lo = state1;
		in[0].hi = pad4;
		state1 = sha256_Transform2(in, state2);
//		state2 = H256;
		in[0].lo = state1;
		in[0].hi = pad4;
		in[0].sf = 0x100;
		res = sha256_Transform2(in, H256);
//		return(swapvec(res));

//		uint8 res = pbkdf_sha256_second2(thread, nonce);
		if (cuda_swab32(res.s7) <= pTarget[7]) {
			uint32_t tmp = atomicExch(&nonceVector[0], nonce);
			
		}


}


void yescrypt_cpu_init(int thr_id, int threads, uint32_t *hash, uint32_t *hash2, uint32_t *hash3, uint32_t *hash4)
{
    
	cudaMemcpyToSymbol(state2, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(sha256test, &hash2, sizeof(hash2), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(prevstate, &hash3, sizeof(hash3), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(B, &hash4, sizeof(hash4), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_YNonce[thr_id], sizeof(uint32_t)); 
	
} 


__host__ uint32_t yescrypt_cpu_hash_k4(int thr_id, int threads, uint32_t startNounce,  int order)
{
	uint32_t result[MAX_GPUS] = {0xffffffff};
	cudaMemset(d_YNonce[thr_id], 0xffffffff, sizeof(uint32_t));

 
	const int threadsperblock = 16;
	const int threadsperblock2 = 16;
	 
 
	dim3 grid2((threads + threadsperblock2 - 1) / threadsperblock2);
	dim3 block2(threadsperblock2);
	
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	
	yescrypt_gpu_hash_k0 << <grid, block >> >(threads, startNounce);
	yescrypt_gpu_hash_k1 << <grid, block >> >(threads, startNounce);
	yescrypt_gpu_hash_k2c << <grid2, block2 >> >(threads, startNounce);
	yescrypt_gpu_hash_k2c1 << <grid2, block2 >> >(threads, startNounce);
	yescrypt_gpu_hash_k5 << <grid, block >> >(threads, startNounce, d_YNonce[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);
	cudaMemcpy(&result[thr_id], d_YNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	
return result[thr_id];
}

__host__ void yescrypt_setBlockTarget(uint32_t* pdata, const void *target)
{

		unsigned char PaddedMessage[128]; //bring balance to the force
		memcpy(PaddedMessage,     pdata, 80);
//		memcpy(PaddedMessage+80, 0, 48);
		uint32_t pad3[16] =
		{
			0x00000000, 0x00000000, 0x00000000, 0x00000000,
			0x00000000, 0x80000000, 0x00000000, 0x00000000,
			0x00000000, 0x00000000, 0x00000000, 0x00000000,
			0x00000000, 0x00000000, 0x00000000, 0x000004a0
		};
		pad3[0] = pdata[16];
		pad3[1] = pdata[17];
		pad3[2] = pdata[18];
//		for (int i = 0; i<10; i++) { printf(" pdata/input %d %08x %08x \n",i,pdata[2*i],pdata[2*i+1]); }
		
		 
		cudaMemcpyToSymbol(shapad, pad3, 16 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, target, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_data, PaddedMessage, 32 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

