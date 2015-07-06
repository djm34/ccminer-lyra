#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
/**
 * __shfl() returns the value of var held by the thread whose ID is given by srcLane.
 * If srcLane is outside the range 0..width-1, the thread's own value of var is returned.
 */
#undef __shfl
#define __shfl(var, srcLane, width) (uint32_t)(var)
#endif

#define merge8(z, x, y, b)\
		z=__byte_perm(x, y, b); \

#define SWAP8(x,y)\
		x=__byte_perm(x, y, 0x5410); \
		y=__byte_perm(x, y, 0x7632);

#define SWAP4(x,y)\
		t = (y<<4); \
		t = (x ^ t); \
		t = 0xf0f0f0f0UL & t; \
		x = (x ^ t); \
		t=  t>>4;\
		y=  y ^ t;

#define SWAP4_final(x,y)\
		t = (y<<4); \
		t = (x ^ t); \
		t = 0xf0f0f0f0UL & t; \
		x = (x ^ t); \


#define SWAP2(x,y)\
		t = (y<<2); \
		t = (x ^ t); \
		t = 0xccccccccUL & t; \
		x = (x ^ t); \
		t=  t>>2;\
		y=  y ^ t;

#define SWAP1(x,y)\
		t = (y+y); \
		t = (x ^ t); \
		t = 0xaaaaaaaaUL & t; \
		x = (x ^ t); \
		t=  t>>1;\
		y=  y ^ t;

__device__ __forceinline__
void to_bitslice_quad(uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
    uint32_t other[8];
	uint32_t t;

	uint32_t perm = (threadIdx.x & 1) ? 0x7362 : 0x5140;
	const unsigned int n = threadIdx.x & 3;
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			input[i] = __shfl((int)input[i], n ^ (3 * (n >= 1 && n <= 2)), 4);
			other[i] = __shfl((int)input[i], (threadIdx.x + 1) & 3, 4);
			input[i] = __shfl((int)input[i], threadIdx.x & 2, 4);
			other[i] = __shfl((int)other[i], threadIdx.x & 2, 4);
		}

		merge8(output[0], input[0], input[4], perm);
		merge8(output[1], other[0], other[4], perm);
		merge8(output[2], input[1], input[5], perm);
		merge8(output[3], other[1], other[5], perm);
		merge8(output[4], input[2], input[6], perm);
		merge8(output[5], other[2], other[6], perm);
		merge8(output[6], input[3], input[7], perm);
		merge8(output[7], other[3], other[7], perm);

		SWAP1(output[0], output[1]);
		SWAP1(output[2], output[3]);
		SWAP1(output[4], output[5]);
		SWAP1(output[6], output[7]);

		SWAP2(output[0], output[2]);
		SWAP2(output[1], output[3]);
		SWAP2(output[4], output[6]);
		SWAP2(output[5], output[7]);

		SWAP4(output[0], output[4]);
		SWAP4(output[1], output[5]);
		SWAP4(output[2], output[6]);
		SWAP4(output[3], output[7]);
}

__device__ __forceinline__
void from_bitslice_quad(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t t;
	const uint32_t perm = 0x7531;//(threadIdx.x & 1) ? 0x3175 : 0x7531;

		output[0] = __byte_perm(input[0], input[4], perm);
		output[2] = __byte_perm(input[1], input[5], perm);
		output[8] = __byte_perm(input[2], input[6], perm);
		output[10] = __byte_perm(input[3], input[7], perm);

		SWAP1(output[0], output[2]);
		SWAP1(output[8], output[10]);

		SWAP2(output[0], output[8]);
		SWAP2(output[2], output[10]);

		output[4] = __byte_perm(output[0], output[8], 0x5410);
		output[8] = __byte_perm(output[0], output[8], 0x7632);
		output[0] = output[4];

		output[6] = __byte_perm(output[2], output[10], 0x5410);
		output[10] = __byte_perm(output[2], output[10], 0x7632);
		output[2] = output[6];

		SWAP4(output[0], output[8]);
		SWAP4(output[2], output[10]);

		if (threadIdx.x & 1)
		{
			output[14] = __byte_perm(output[10], 0, 0x3232);
			output[12] = __byte_perm(output[8], 0, 0x3232);
			output[6] = __byte_perm(output[2], 0, 0x3232);
			output[4] = __byte_perm(output[0], 0, 0x3232);

			output[0] = __byte_perm(output[0], 0, 0x1032);
			output[2] = __byte_perm(output[2], 0, 0x1032);
			output[8] = __byte_perm(output[8], 0, 0x1032);
			output[10] = __byte_perm(output[10], 0, 0x1032);
		}
		else
		{
			output[4] = output[0];
			output[6] = output[2];
			output[12] = output[8];
			output[14] = output[10];
		}

	output[0] = __byte_perm(output[0], __shfl((int)output[0], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[2] = __byte_perm(output[2], __shfl((int)output[2], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[4] = __byte_perm(output[4], __shfl((int)output[4], (threadIdx.x + 1) & 3, 4), 0x7632);
	output[6] = __byte_perm(output[6], __shfl((int)output[6], (threadIdx.x + 1) & 3, 4), 0x7632);
	output[8] = __byte_perm(output[8], __shfl((int)output[8], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[10] = __byte_perm(output[10], __shfl((int)output[10], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[12] = __byte_perm(output[12], __shfl((int)output[12], (threadIdx.x + 1) & 3, 4), 0x7632);
	output[14] = __byte_perm(output[14], __shfl((int)output[14], (threadIdx.x + 1) & 3, 4), 0x7632);

	output[0 + 1] = __shfl((int)output[0], (threadIdx.x + 2) & 3, 4);
	output[2 + 1] = __shfl((int)output[2], (threadIdx.x + 2) & 3, 4);
	output[4 + 1] = __shfl((int)output[4], (threadIdx.x + 2) & 3, 4);
	output[6 + 1] = __shfl((int)output[6], (threadIdx.x + 2) & 3, 4);
	output[8 + 1] = __shfl((int)output[8], (threadIdx.x + 2) & 3, 4);
	output[10 + 1] = __shfl((int)output[10], (threadIdx.x + 2) & 3, 4);
	output[12 + 1] = __shfl((int)output[12], (threadIdx.x + 2) & 3, 4);
	output[14 + 1] = __shfl((int)output[14], (threadIdx.x + 2) & 3, 4);

}

__device__ __forceinline__
void from_bitslice_quad_final(const uint32_t *const __restrict__ input, uint32_t *const __restrict__ output)
{
	uint32_t t;
	const uint32_t perm = 0x7531;//(threadIdx.x & 1) ? 0x3175 : 0x7531;

	if (threadIdx.x & 3)
	{

		output[0] = __byte_perm(input[0], input[4], perm);
		output[2] = __byte_perm(input[1], input[5], perm);
		output[8] = __byte_perm(input[2], input[6], perm);
		output[10] = __byte_perm(input[3], input[7], perm);
		SWAP1(output[0], output[2]);
		SWAP1(output[8], output[10]);
		SWAP2(output[2], output[10]);
		output[6] = __byte_perm(output[2], output[10], 0x5410);
		output[10] = __byte_perm(output[2], output[10], 0x7632);
		SWAP4_final(output[6], output[10]);
		output[6] = __byte_perm(output[6], 0, 0x3232);
	}
	else
	{
		output[0] = __byte_perm(input[0], input[4], perm);
		output[2] = __byte_perm(input[1], input[5], perm);
		output[8] = __byte_perm(input[2], input[6], perm);
		output[10] = __byte_perm(input[3], input[7], perm);

		SWAP1(output[0], output[2]);
		SWAP1(output[8], output[10]);

		SWAP2(output[0], output[8]);
		SWAP2(output[2], output[10]);

		output[4] = __byte_perm(output[0], output[8], 0x5410);
		output[8] = __byte_perm(output[0], output[8], 0x7632);
		output[0] = output[4];

		output[6] = __byte_perm(output[2], output[10], 0x5410);
		output[10] = __byte_perm(output[2], output[10], 0x7632);
		output[2] = output[6];

		SWAP4(output[0], output[8]);
		SWAP4(output[2], output[10]);

		if (threadIdx.x & 1)
		{
			output[14] = __byte_perm(output[10], 0, 0x3232);
			output[12] = __byte_perm(output[8], 0, 0x3232);
			output[6] = __byte_perm(output[2], 0, 0x3232);
			output[4] = __byte_perm(output[0], 0, 0x3232);

			output[0] = __byte_perm(output[0], 0, 0x1032);
			output[2] = __byte_perm(output[2], 0, 0x1032);
			output[8] = __byte_perm(output[8], 0, 0x1032);
			output[10] = __byte_perm(output[10], 0, 0x1032);
		}
		else
		{
			output[4] = output[0];
			output[6] = output[2];
			output[12] = output[8];
			output[14] = output[10];
		}
	}


	output[0] = __byte_perm(output[0], __shfl((int)output[0], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[2] = __byte_perm(output[2], __shfl((int)output[2], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[4] = __byte_perm(output[4], __shfl((int)output[4], (threadIdx.x + 1) & 3, 4), 0x7632);
	output[6] = __byte_perm(output[6], __shfl((int)output[6], (threadIdx.x + 1) & 3, 4), 0x7632);
	output[8] = __byte_perm(output[8], __shfl((int)output[8], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[10] = __byte_perm(output[10], __shfl((int)output[10], (threadIdx.x + 1) & 3, 4), 0x7610);
	output[12] = __byte_perm(output[12], __shfl((int)output[12], (threadIdx.x + 1) & 3, 4), 0x7632);
	output[14] = __byte_perm(output[14], __shfl((int)output[14], (threadIdx.x + 1) & 3, 4), 0x7632);

	output[0 + 1] = __shfl((int)output[0], (threadIdx.x + 2) & 3, 4);
	output[2 + 1] = __shfl((int)output[2], (threadIdx.x + 2) & 3, 4);
	output[4 + 1] = __shfl((int)output[4], (threadIdx.x + 2) & 3, 4);
	output[6 + 1] = __shfl((int)output[6], (threadIdx.x + 2) & 3, 4);
	output[8 + 1] = __shfl((int)output[8], (threadIdx.x + 2) & 3, 4);
	output[10 + 1] = __shfl((int)output[10], (threadIdx.x + 2) & 3, 4);
	output[12 + 1] = __shfl((int)output[12], (threadIdx.x + 2) & 3, 4);
	output[14 + 1] = __shfl((int)output[14], (threadIdx.x + 2) & 3, 4);

}
