

#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"
#define TPB 16
//

 
#define Nrow 4
#define Ncol 4
#if __CUDA_ARCH__ < 500 
#define vectype ulonglong4
#define u64type uint64_t
#define memshift 4
#elif __CUDA_ARCH__ == 500
#define u64type uint2
#define vectype uint28
#define memshift 3
#else 
#define u64type uint2
#define vectype uint28
#define memshift 3   
#endif 
__device__ vectype  *DMatrix;

 
static __device__ __forceinline__ void Gfunc_v35(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
	   
	a += b; d ^= a; d = SWAPDWORDS2(d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);

}

static __device__ __forceinline__ void Gfunc_v35_p1(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{

	a += b; d ^= a; d = SWAPDWORDS2(d);
	c += d; b ^= c; b = ROR24(b);
}

static __device__ __forceinline__ void Gfunc_v35_p2(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
}


static __device__ __forceinline__ void Gfunc_v35(uint64_t & a, uint64_t &b, uint64_t &c, uint64_t &d)
{

	a += b; d ^= a; d = ROTR64(d, 32);
	c += d; b ^= c; b = ROTR64(b, 24);
	a += b; d ^= a; d = ROTR64(d, 16);
	c += d; b ^= c; b = ROTR64(b, 63);

}

static __device__ __forceinline__ void Gfunc_v35_p1(uint64_t & a, uint64_t &b, uint64_t &c, uint64_t &d)
{

	a += b; d ^= a; d = ROTR64(d, 32);
	c += d; b ^= c; b = ROTR64(b, 24);
}

static __device__ __forceinline__ void Gfunc_v35_p2(uint64_t & a, uint64_t &b, uint64_t &c, uint64_t &d)
{

	a += b; d ^= a; d = ROTR64(d, 16);
	c += d; b ^= c; b = ROTR64(b, 63);
}

#define RORa(d) make_uint28(SWAPDWORDS2(d.x),SWAPDWORDS2(d.y),SWAPDWORDS2(d.z),SWAPDWORDS2(d.w))
#define RORb(d) make_uint28(ROR24(d.x),ROR24(d.y),ROR24(d.z),ROR24(d.w))
#define RORc(d) make_uint28(ROR16(d.x),ROR16(d.y),ROR16(d.z),ROR16(d.w))
#define RORd(d) make_uint28(ROR2(d.x,63),ROR2(d.y,63),ROR2(d.z,63),ROR2(d.w,63))


static __device__ __forceinline__ ulonglong4 make_vectype(const uint64_t  a, const uint64_t b, const uint64_t c, const uint64_t d)
{
	return make_ulonglong4(a, b, c, d);	
}

static __device__ __forceinline__ uint28 make_vectype(const uint2  a, const uint2 b, const uint2 c, const uint2 d)
{
	return make_uint28(a, b, c, d);
}


static __device__ __forceinline__ void Gfunc_v4(ulonglong4 & a, ulonglong4 &b, ulonglong4 &c, ulonglong4 &d)
{
#define ROR4(d,n) make_ulonglong4(ROTR64(d.x,n),ROTR64(d.y,n),ROTR64(d.z,n),ROTR64(d.w,n))
	a += b; d ^= a; d = ROR4(d, 32);
	c += d; b ^= c; b = ROR4(b, 24);
	a += b; d ^= a; d = ROR4(d, 16);
	c += d; b ^= c; b = ROR4(b, 63);
#undef ROR4
}

static __device__ __forceinline__ void Gfunc_v4(uint28 & a, uint28 &b, uint28 &c, uint28 &d)
{
#define ROR4(d,n) make_uint28(ROR2(d.x,n),ROR2(d.y,n),ROR2(d.z,n),ROR2(d.w,n))
	a += b; d ^= a; d = RORa(d);
	c += d; b ^= c; b = RORb(b);
	a += b; d ^= a; d = RORc(d);
	c += d; b ^= c; b = RORd(b);
#undef ROR4
}



static __device__ __forceinline__ void round_lyra64(uint64_t* s)  
{  
	Gfunc_v35(s[0], s[4], s[8], s[12]);  
	Gfunc_v35(s[1], s[5], s[9], s[13]);  
	Gfunc_v35(s[2], s[6], s[10], s[14]);  
	Gfunc_v35(s[3], s[7], s[11], s[15]);  
	Gfunc_v35(s[0], s[5], s[10], s[15]);   
	Gfunc_v35(s[1], s[6], s[11], s[12]);  
	Gfunc_v35(s[2], s[7], s[8], s[13]);  
	Gfunc_v35(s[3], s[4], s[9], s[14]);  
}

static __device__ __forceinline__ void round_lyra_v35(uint2_16* s)
{
	    Gfunc_v35(s[0].s0, s[0].s4, s[0].s8, s[0].sc);  
		Gfunc_v35(s[0].s1, s[0].s5, s[0].s9, s[0].sd);  
		Gfunc_v35(s[0].s2, s[0].s6, s[0].sa, s[0].se);  
		Gfunc_v35(s[0].s3, s[0].s7, s[0].sb, s[0].sf);  
		Gfunc_v35(s[0].s0, s[0].s5, s[0].sa, s[0].sf);  
		Gfunc_v35(s[0].s1, s[0].s6, s[0].sb, s[0].sc);  
		Gfunc_v35(s[0].s2, s[0].s7, s[0].s8, s[0].sd);  
		Gfunc_v35(s[0].s3, s[0].s4, s[0].s9, s[0].se);  
}

static __device__ __forceinline__ void round_lyra_v35(uint2* s)
{
	Gfunc_v35(s[0], s[4], s[8], s[12]);
	Gfunc_v35(s[1], s[5], s[9], s[13]);
	Gfunc_v35(s[2], s[6], s[10], s[14]);
	Gfunc_v35(s[3], s[7], s[11], s[15]);
	Gfunc_v35(s[0], s[5], s[10], s[15]);
	Gfunc_v35(s[1], s[6], s[11], s[12]);
	Gfunc_v35(s[2], s[7], s[8], s[13]);
	Gfunc_v35(s[3], s[4], s[9], s[14]);
}

static __device__ __forceinline__ void round_lyra_v35(vectype* s)
{

	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);

}
 
static __device__ __forceinline__ void reduceDuplex(vectype state[4], uint32_t thread)
{


	    vectype state1[3]; 
		uint32_t ps1 = (Nrow * Ncol * memshift * thread);
		uint32_t ps2 = (memshift * (Ncol-1) + memshift * Ncol + Nrow * Ncol * memshift * thread);

#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
        uint32_t s1 = ps1 + i*memshift;
        uint32_t s2 = ps2 - i*memshift;  
		
		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix+s1)[j]); 
 
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];
		round_lyra_v35(state); 
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state1[j];

	}

}

static __device__ __forceinline__ void reduceDuplexV3(vectype state[4], uint32_t thread)
{


	vectype state1[3];
	uint32_t ps1 = (Nrow * Ncol * memshift * thread);
//                     colomn             row
	uint32_t ps2 = (memshift * (Ncol - 1) * Nrow + memshift * 1 + Nrow * Ncol * memshift * thread);

#pragma unroll 4
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + Nrow * i *memshift;
		uint32_t s2 = ps2 - Nrow * i *memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);

		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];
		round_lyra_v35(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];


		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state1[j];

	}

}

static __device__ __forceinline__ void reduceDuplexRowSetupV2(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{


		vectype state2[3],state1[3];

		uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
		uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
		uint32_t ps3 = (memshift * (Ncol-1) + memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);


//#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + i*memshift;
		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 - i*memshift;

		for (int j = 0; j < 3; j++) 
			state1[j]= __ldg4(&(DMatrix + s1)[j]);
		for (int j = 0; j < 3; j++)
			state2[j]= __ldg4(&(DMatrix + s2)[j]);
		for (int j = 0; j < 3; j++) {
			vectype tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}
		

		round_lyra_v35(state);

		for (int j = 0; j < 3; j++) {
			state1[j] ^= state[j];
			(DMatrix + s3)[j] = state1[j];
		}
 
		   ((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++) 
			((uint2*)state2)[j+1] ^= ((uint2*)state)[j];



		for (int j = 0; j < 3; j++)
		    (DMatrix + s2)[j] = state2[j];
		
	}


}

static __device__ __forceinline__ void reduceDuplexRowSetupV3(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{


	vectype state2[3], state1[3];
	
	uint32_t ps1 = (memshift *  rowIn                     + Nrow * Ncol * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut                   + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (Nrow * memshift * (Ncol - 1) + memshift *  rowOut + Nrow * Ncol * memshift * thread);

	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + Nrow*i*memshift;
		uint32_t s2 = ps2 + Nrow*i*memshift;
		uint32_t s3 = ps3 - Nrow*i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1 )[j]);
		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2 )[j]);
		for (int j = 0; j < 3; j++) {
			vectype tmp = state1[j] + state2[j];
			state[j] ^= tmp;
		}


		round_lyra_v35(state);

		for (int j = 0; j < 3; j++) {
			state1[j] ^= state[j];
			(DMatrix + s3)[j] = state1[j];
		}

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];



		for (int j = 0; j < 3; j++)
			(DMatrix + s2)[j] = state2[j];

	}


}


static __device__ __forceinline__ void reduceDuplexRowtV2(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{

		vectype state1[3],state2[3];
		uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
		uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
		uint32_t ps3 = (memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);

//#pragma unroll 1
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + i*memshift;
		uint32_t s2 = ps2 + i*memshift;
		uint32_t s3 = ps3 + i*memshift;


		for (int j = 0; j < 3; j++)  
			state1[j] = __ldg4(&(DMatrix + s1)[j]);


		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);


		for (int j = 0; j < 3; j++)
			          state1[j] += state2[j];

		for (int j = 0; j < 3; j++)
			          state[j] ^= state1[j];


		round_lyra_v35(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++)
		((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

if (rowInOut != rowOut) {

	for (int j = 0; j < 3; j++)
		(DMatrix + s2)[j] = state2[j];

	for (int j = 0; j < 3; j++)
		(DMatrix + s3)[j] ^= state[j];

} else {

	for (int j = 0; j < 3; j++)
		state2[j] ^= state[j];

	for (int j = 0; j < 3; j++)
		(DMatrix + s2)[j]=state2[j];
}






	}
}

static __device__ __forceinline__ void reduceDuplexRowtV3(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{

	vectype state1[3], state2[3];
	uint32_t ps1 = (memshift * rowIn + Nrow * Ncol * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut + Nrow * Ncol * memshift * thread);
	uint32_t ps3 = (memshift * rowOut + Nrow * Ncol * memshift * thread);

#pragma nounroll 
	for (int i = 0; i < Ncol; i++)
	{
		uint32_t s1 = ps1 + Nrow * i*memshift;
		uint32_t s2 = ps2 + Nrow * i*memshift;
		uint32_t s3 = ps3 + Nrow * i*memshift;


		for (int j = 0; j < 3; j++)
			state1[j] = __ldg4(&(DMatrix + s1)[j]);


		for (int j = 0; j < 3; j++)
			state2[j] = __ldg4(&(DMatrix + s2)[j]);


		for (int j = 0; j < 3; j++)
			state1[j] += state2[j];

		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];


		round_lyra_v35(state);

		((uint2*)state2)[0] ^= ((uint2*)state)[11];
		for (int j = 0; j < 11; j++)
			((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

		if (rowInOut != rowOut) {

			for (int j = 0; j < 3; j++)
				(DMatrix + s2)[j] = state2[j];

			for (int j = 0; j < 3; j++)
				(DMatrix + s3)[j] ^= state[j];

		}
		else {

			for (int j = 0; j < 3; j++)
				state2[j] ^= state[j];

			for (int j = 0; j < 3; j++)
				(DMatrix + s2)[j] = state2[j];
		}






	}
}


#if __CUDA_ARCH__ < 500
__global__	__launch_bounds__(128, 1)
#elif __CUDA_ARCH__ == 500
__global__	__launch_bounds__(16, 1)
#else
__global__	__launch_bounds__(TPB, 1)
#endif
void lyra2v2_gpu_hash_32_v3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	vectype state[4];


	uint28 blake2b_IV[2];
	uint28 padding[2];
	if (threadIdx.x == 0) {

		((uint2_8*)blake2b_IV)[0] = {
			{ 0xf3bcc908, 0x6a09e667 },
			{ 0x84caa73b, 0xbb67ae85 },
			{ 0xfe94f82b, 0x3c6ef372 },
			{ 0x5f1d36f1, 0xa54ff53a },
			{ 0xade682d1, 0x510e527f },
			{ 0x2b3e6c1f, 0x9b05688c },
			{ 0xfb41bd6b, 0x1f83d9ab },
			{ 0x137e2179, 0x5be0cd19 }
		};
		((uint2_8*)padding)[0] = {
			{ 0x20, 0x0 },
			{ 0x20, 0x0 },
			{ 0x20, 0x0 },
			{ 0x01, 0x0 },
			{ 0x04, 0x0 },
			{ 0x04, 0x0 },
			{ 0x80, 0x0 },
			{ 0x0, 0x01000000 }
		};

	}

#if __CUDA_ARCH__ == 350
	if (thread < threads)
#endif
	{

		((uint2*)state)[0] = __ldg(&outputHash[thread]);
		((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
		((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);
		state[1] = state[0];
		state[2] = shuffle4(((vectype*)blake2b_IV)[0], 0);
		state[3] = shuffle4(((vectype*)blake2b_IV)[1], 0);

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);
		state[0] ^= shuffle4(((vectype*)padding)[0], 0);
		state[1] ^= shuffle4(((vectype*)padding)[1], 0);


		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		uint32_t ps1 = (4 * memshift * 3 + 16 * memshift * thread);

		//#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			uint32_t s1 = ps1 - 4 * memshift * i;
			for (int j = 0; j < 3; j++)
				(DMatrix + s1)[j] = (state)[j];

			round_lyra_v35(state);
		}

		reduceDuplexV3(state, thread);
		reduceDuplexRowSetupV3(1, 0, 2, state, thread);
		reduceDuplexRowSetupV3(2, 1, 3, state, thread);

		uint32_t rowa;
		int prev = 3;
		for (int i = 0; i < 4; i++)
		{
			rowa = ((uint2*)state)[0].x & 3;  reduceDuplexRowtV3(prev, rowa, i, state, thread);
			prev = i;
		}

		uint32_t shift = (memshift * rowa + 16 * memshift * thread);

		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);


		outputHash[thread] = ((uint2*)state)[0];
		outputHash[thread + threads] = ((uint2*)state)[1];
		outputHash[thread + 2 * threads] = ((uint2*)state)[2];
		outputHash[thread + 3 * threads] = ((uint2*)state)[3];
		//		((vectype*)outputHash)[thread] = state[0];

	} //thread
}



#if __CUDA_ARCH__ < 500
__global__	__launch_bounds__(64, 1)
#elif __CUDA_ARCH__ == 500
__global__	__launch_bounds__(32, 1)
#else
__global__	__launch_bounds__(TPB, 1)
#endif
void lyra2v2_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	   vectype state[4];

	   uint28 blake2b_IV[2];
	   uint28 padding[2];
	   if (threadIdx.x == 0) {

		   ((uint2_8*)blake2b_IV)[0] = {
			   { 0xf3bcc908, 0x6a09e667 },
			   { 0x84caa73b, 0xbb67ae85 },
			   { 0xfe94f82b, 0x3c6ef372 },
			   { 0x5f1d36f1, 0xa54ff53a },
			   { 0xade682d1, 0x510e527f },
			   { 0x2b3e6c1f, 0x9b05688c },
			   { 0xfb41bd6b, 0x1f83d9ab },
			   { 0x137e2179, 0x5be0cd19 }
		   };
		   ((uint2_8*)padding)[0] = {
			   { 0x20, 0x0 },
			   { 0x20, 0x0 },
			   { 0x20, 0x0 },
			   { 0x01, 0x0 },
			   { 0x04, 0x0 },
			   { 0x04, 0x0 },
			   { 0x80, 0x0 },
			   { 0x0, 0x01000000 }
		   };
       }

#if __CUDA_ARCH__ == 350
	if (thread < threads)
#endif
	{
 
		 ((uint2*)state)[0] = __ldg(&outputHash[thread]);
		 ((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		 ((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
		 ((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);

		 state[1] = state[0];

		 state[2] = shuffle4(((vectype*)blake2b_IV)[0], 0);
		 state[3] = shuffle4(((vectype*)blake2b_IV)[1], 0);

		 for (int i = 0; i<12; i++)
			 round_lyra_v35(state);
		 state[0] ^= shuffle4(((vectype*)padding)[0], 0);
		 state[1] ^= shuffle4(((vectype*)padding)[1], 0);

		 for (int i = 0; i<12; i++)
			 round_lyra_v35(state);

		uint32_t ps1 = (memshift * (Ncol - 1) + Nrow * Ncol * memshift * thread);

		for (int i = 0; i < Ncol; i++)
		{
			uint32_t s1 = ps1 - memshift * i;
			for (int j = 0; j < 3; j++)
			    (DMatrix + s1)[j] = (state)[j];

			round_lyra_v35(state);
		}


		reduceDuplex(state, thread);

		reduceDuplexRowSetupV2(1, 0, 2, state,  thread);
		reduceDuplexRowSetupV2(2, 1, 3, state,  thread);
uint32_t rowa;
int prev=3;

         for (int i = 0; i < 4; i++)
        {
	     rowa = ((uint2*)state)[0].x & 3;  reduceDuplexRowtV2(prev, rowa, i, state, thread);
         prev=i;
        }


		uint32_t shift = (memshift * Ncol * rowa + Nrow * Ncol * memshift * thread);

		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
        			round_lyra_v35(state);
		

		outputHash[thread]=            ((uint2*)state)[0];
		outputHash[thread + threads] = ((uint2*)state)[1];
		outputHash[thread + 2 * threads] = ((uint2*)state)[2]; 
		outputHash[thread + 3 * threads] = ((uint2*)state)[3];
//		((vectype*)outputHash)[thread] = state[0];

	} //thread
}


__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads,uint64_t *hash)
{
	cudaMemcpyToSymbol(DMatrix, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
}



__host__ 
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash)
{
uint32_t tpb;
	if (device_sm[device_map[thr_id]]<500) 
      tpb = 64;
	else if (device_sm[device_map[thr_id]]==500)
      tpb = 32; 
    else 
      tpb = TPB;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	if (device_sm[device_map[thr_id]] >= 500)
		lyra2v2_gpu_hash_32 << <grid, block >> > (threads, startNounce, (uint2*)d_outputHash);
    else 
    	lyra2v2_gpu_hash_32_v3 <<<grid, block>>> (threads, startNounce,(uint2*) d_outputHash);

	//MyStreamSynchronize(NULL, order, thr_id);
}

  