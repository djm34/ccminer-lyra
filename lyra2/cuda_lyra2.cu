

#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"
#define TPB 8
//



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
#define memshift 4   
#endif 
__device__ vectype  *DMatrix;

 
static __device__ __forceinline__ void Gfunc_v35(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{

	a += b; d ^= a; d = SWAPDWORDS2(d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);

}

static __device__ __forceinline__ void Gfunc_v35(unsigned long long & a, unsigned long long &b, unsigned long long &c, unsigned long long &d)
{

	a += b; d ^= a; d = ROTR64(d, 32);
	c += d; b ^= c; b = ROTR64(b, 24);
	a += b; d ^= a; d = ROTR64(d, 16);
	c += d; b ^= c; b = ROTR64(b, 63);

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



__device__ __forceinline__ void reduceDuplex(vectype state[4], uint32_t thread)
{


	    vectype state1[3]; 
		uint32_t ps1 = (256 * thread);
		uint32_t ps2 = (memshift * 7 + memshift * 8 + 256 * thread);

#pragma unroll 4
	for (int i = 0; i < 8; i++)
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

__device__ __forceinline__ void reduceDuplexV3(vectype state[4], uint32_t thread)
{


	vectype state1[3];
	uint32_t ps1 = (256 * thread);
//                     colomn             row
	uint32_t ps2 = (memshift * 7 * 8 + memshift * 1 + 64 * memshift * thread);

#pragma unroll 4
	for (int i = 0; i < 8; i++)
	{
		uint32_t s1 = ps1 + 8 * i *memshift;
		uint32_t s2 = ps2 - 8 * i *memshift;

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

__device__ __forceinline__ void reduceDuplexRowSetupV2(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{


		vectype state2[3],state1[3];

		uint32_t ps1 = (              memshift * 8 * rowIn    + 256 * thread);
		uint32_t ps2 = (              memshift * 8 * rowInOut + 256 * thread);
		uint32_t ps3 = (memshift*7  + memshift * 8 * rowOut   + 256 * thread);


#pragma unroll 1
	for (int i = 0; i < 8; i++)
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

__device__ __forceinline__ void reduceDuplexRowSetupV3(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{


	vectype state2[3], state1[3];
	
	uint32_t ps1 = (                  memshift *  rowIn    + 64 * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut +                    64 * memshift* thread);
	uint32_t ps3 = (8 * memshift * 7 + memshift *  rowOut +  64 * memshift * thread);
	/*
	uint32_t ps1 = (256 * thread);
	uint32_t ps2 = (256 * thread);
	uint32_t ps3 = (256 * thread);
    */
#pragma nounroll 
	for (int i = 0; i < 8; i++)
	{
		uint32_t s1 = ps1 + 8*i*memshift;
		uint32_t s2 = ps2 + 8*i*memshift;
		uint32_t s3 = ps3 - 8*i*memshift;

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


__device__ __forceinline__ void reduceDuplexRowtV2(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{

		vectype state1[3],state2[3];
		uint32_t ps1 = (memshift * 8 * rowIn + 256 * thread);
		uint32_t ps2 = (memshift * 8 * rowInOut + 256 * thread);
		uint32_t ps3 = (memshift * 8 * rowOut + 256 * thread);

#pragma unroll 1
	for (int i = 0; i < 8; i++)
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

__device__ __forceinline__ void reduceDuplexRowtV3(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{

	vectype state1[3], state2[3];
	uint32_t ps1 = (memshift * rowIn + 64 * memshift * thread);
	uint32_t ps2 = (memshift * rowInOut + 64 * memshift * thread);
	uint32_t ps3 = (memshift * rowOut + 64 *memshift * thread);

#pragma nounroll 
	for (int i = 0; i < 8; i++)
	{
		uint32_t s1 = ps1 + 8 * i*memshift;
		uint32_t s2 = ps2 + 8 * i*memshift;
		uint32_t s3 = ps3 + 8 * i*memshift;


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
__global__	__launch_bounds__(48, 1)
#elif __CUDA_ARCH__ == 500
__global__	__launch_bounds__(16, 1)
#else
__global__	__launch_bounds__(TPB, 1)
#endif
void lyra2_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	   vectype state[4];
#if __CUDA_ARCH__ > 350
	const uint28 blake2b_IV[2] = {
		{{ 0xf3bcc908, 0x6a09e667 },
		{ 0x84caa73b, 0xbb67ae85 },
		{ 0xfe94f82b, 0x3c6ef372 },
		{ 0x5f1d36f1, 0xa54ff53a }},
		{{ 0xade682d1, 0x510e527f },
		{ 0x2b3e6c1f, 0x9b05688c },
		{ 0xfb41bd6b, 0x1f83d9ab },
		{ 0x137e2179, 0x5be0cd19 }}};
#else 
		const ulonglong4 blake2b_IV[2] = {
			{ 0x6a09e667f3bcc908,  
			  0xbb67ae8584caa73b,  
			  0x3c6ef372fe94f82b,  
			  0xa54ff53a5f1d36f1   },
			{ 0x510e527fade682d1,  
			  0x9b05688c2b3e6c1f,  
			  0x1f83d9abfb41bd6b,  
			  0x5be0cd19137e2179  } };
#endif
	
#if __CUDA_ARCH__ == 350
	if (thread < threads)
#endif
	{
 
		 ((uint2*)state)[0] = __ldg(&outputHash[thread]);
		 ((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		 ((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
		 ((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);
//		 state[0] = __ldg4(&((vectype*)outputHash)[thread]);
		 state[1] = state[0];
		 state[2] = ((vectype*)blake2b_IV)[0];
		 state[3] = ((vectype*)blake2b_IV)[1];

 
		for (int i = 0; i<24; i++) { round_lyra_v35(state); } //because 12 is not enough

             uint32_t ps1 = (memshift * 7  + 256 * thread);

		for (int i = 0; i < 8; i++)
		{
			uint32_t s1 = ps1 - memshift * i;
			for (int j = 0; j < 3; j++)
			    (DMatrix + s1)[j] = (state)[j];

			round_lyra_v35(state);
		}


		reduceDuplex(state, thread);

		reduceDuplexRowSetupV2(1, 0, 2, state,  thread);
		reduceDuplexRowSetupV2(2, 1, 3, state,  thread);
		reduceDuplexRowSetupV2(3, 0, 4, state,  thread);
		reduceDuplexRowSetupV2(4, 3, 5, state,  thread);
		reduceDuplexRowSetupV2(5, 2, 6, state,  thread);
		reduceDuplexRowSetupV2(6, 1, 7, state,  thread);
		uint32_t rowa = ((uint2*)state)[0].x & 7;

		reduceDuplexRowtV2(7, rowa, 0, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(0, rowa, 3, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(3, rowa, 6, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(6, rowa, 1, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(1, rowa, 4, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(4, rowa, 7, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(7, rowa, 2, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV2(2, rowa, 5, state, thread);

		uint32_t shift = (memshift * 8 * rowa + 256 * thread);

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

#if __CUDA_ARCH__ < 500
__global__	__launch_bounds__(48, 1)
#elif __CUDA_ARCH__ == 500
__global__	__launch_bounds__(16, 1)
#else
__global__	__launch_bounds__(TPB, 1)
#endif
void lyra2_gpu_hash_32_v3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	vectype state[4];

#if __CUDA_ARCH__ > 350
	const uint28 blake2b_IV[2] = {
		{ { 0xf3bcc908, 0x6a09e667 },
		{ 0x84caa73b, 0xbb67ae85 },
		{ 0xfe94f82b, 0x3c6ef372 },
		{ 0x5f1d36f1, 0xa54ff53a } },
		{ { 0xade682d1, 0x510e527f },
		{ 0x2b3e6c1f, 0x9b05688c },
		{ 0xfb41bd6b, 0x1f83d9ab },
		{ 0x137e2179, 0x5be0cd19 } } };
#else 
	const ulonglong4 blake2b_IV[2] = {
		{ 0x6a09e667f3bcc908,
		0xbb67ae8584caa73b,
		0x3c6ef372fe94f82b,
		0xa54ff53a5f1d36f1 },
		{ 0x510e527fade682d1,
		0x9b05688c2b3e6c1f,
		0x1f83d9abfb41bd6b,
		0x5be0cd19137e2179 } };
#endif


#if __CUDA_ARCH__ == 350
	if (thread < threads)
#endif
	{

		((uint2*)state)[0] = __ldg(&outputHash[thread]);
		((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
		((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
		((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);
		
		state[1] = state[0];

		state[2] = ((vectype*)blake2b_IV)[0];
		state[3] = ((vectype*)blake2b_IV)[1];

		for (int i = 0; i<24; i++) 
                round_lyra_v35(state);  //because 12 is not enough

		uint32_t ps1 = (8 * memshift * 7 + 64 * memshift * thread);


		for (int i = 0; i < 8; i++)
		{
			uint32_t s1 = ps1 - 8 * memshift * i;
			for (int j = 0; j < 3; j++)
				(DMatrix + s1)[j] = (state)[j];

			round_lyra_v35(state);
		}


		reduceDuplexV3(state, thread);

		reduceDuplexRowSetupV3(1, 0, 2, state, thread);
		reduceDuplexRowSetupV3(2, 1, 3, state, thread);
		reduceDuplexRowSetupV3(3, 0, 4, state, thread);
		reduceDuplexRowSetupV3(4, 3, 5, state, thread);
		reduceDuplexRowSetupV3(5, 2, 6, state, thread);
		reduceDuplexRowSetupV3(6, 1, 7, state, thread);
		uint32_t rowa = ((uint2*)state)[0].x & 7;

		reduceDuplexRowtV3(7, rowa, 0, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(0, rowa, 3, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(3, rowa, 6, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(6, rowa, 1, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(1, rowa, 4, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(4, rowa, 7, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(7, rowa, 2, state, thread);
		rowa = ((uint2*)state)[0].x & 7;
		reduceDuplexRowtV3(2, rowa, 5, state, thread);

		uint32_t shift = (memshift * rowa + 64 * memshift * thread);

		for (int j = 0; j < 3; j++)
			state[j] ^= __ldg4(&(DMatrix + shift)[j]);

		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);


		outputHash[thread] = ((uint2*)state)[0];
		outputHash[thread + threads] = ((uint2*)state)[1];
		outputHash[thread + 2 * threads] = ((uint2*)state)[2];
		outputHash[thread + 3 * threads] = ((uint2*)state)[3];
		
	} //thread
}




__host__
void lyra2_cpu_init(int thr_id, uint32_t threads,uint64_t *hash)
{
	cudaMemcpyToSymbol(DMatrix, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
}



__host__ 
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash)
{
uint32_t tpb;
	if (device_sm[device_map[thr_id]]<500) 
      tpb = 48;
	else if (device_sm[device_map[thr_id]]==500)
      tpb = 16; 
    else 
      tpb = TPB;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);

	if (device_sm[device_map[thr_id]] == 500)
		lyra2_gpu_hash_32 << <grid, block >> > (threads, startNounce, (uint2*)d_outputHash);
    else 
    	lyra2_gpu_hash_32_v3 <<<grid, block>>> (threads, startNounce,(uint2*) d_outputHash);


}

  