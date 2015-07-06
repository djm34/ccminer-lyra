#include "cuda_helper.h"


#ifdef _MSC_VER
#define UINT2(x,y) { x, y }
#else
#define UINT2(x,y) (uint2) { x, y }
#endif

__constant__ static __align__(16) uint32_t c_E8_bslice32[42][8] = {
	// Round 0 (Function0)
		{ 0xa2ded572, 0x90d6ab81, 0x67f815df, 0xf6875a4d, 0x0a15847b, 0xc54f9f4e, 0x571523b7, 0x402bd1c3 },
		{ 0xe03a98ea, 0xb4960266, 0x9cfa455c, 0x8a53bbf2, 0x99d2c503, 0x1a1456b5, 0x9a99b266, 0x31a2db88 }, // 1
		{ 0x5c5aa303, 0x8019051c, 0xdb0e199a, 0x1d959e84, 0x0ab23f40, 0xadeb336f, 0x1044c187, 0xdccde75e }, // 2
		{ 0x9213ba10, 0x39812c0a, 0x416bbf02, 0x5078aa37, 0x156578dc, 0xd2bf1a3f, 0xd027bbf7, 0xd3910041 }, // 3
		{ 0x0d5a2d42, 0x0ba75c18, 0x907eccf6, 0xac442bc7, 0x9c9f62dd, 0xd665dfd1, 0xce97c092, 0x23fcc663 }, // 4
		{ 0x036c6e97, 0xbb03f1ee, 0x1ab8e09e, 0xfa618e5d, 0x7e450521, 0xb29796fd, 0xa8ec6c44, 0x97818394 }, // 5
		{ 0x37858e4a, 0x8173fe8a, 0x2f3003db, 0x6c69b8f8, 0x2d8d672a, 0x4672c78a, 0x956a9ffb, 0x14427fc0 }, // 6
		// Round 7 (Function0)
		{ 0x8f15f4c5, 0xb775de52, 0xc45ec7bd, 0xbc88e4ae, 0xa76f4475, 0x1e00b882, 0x80bb118f, 0xf4a3a698 },
		{ 0x338ff48e, 0x20edf1b6, 0x1563a3a9, 0xfde05a7c, 0x24565faa, 0x5ae9ca36, 0x89f9b7d5, 0x362c4206 },
		{ 0x433529ce, 0x591ff5d0, 0x3d98fe4e, 0x86814e6f, 0x74f93a53, 0x81ad9d0e, 0xa74b9a73, 0x9f5ad8af },
		{ 0x670605a7, 0x26077447, 0x6a6234ee, 0x3f1080c6, 0xbe280b8b, 0x6f7ea0e0, 0x2717b96e, 0x7b487ec6 },
		{ 0xa50a550d, 0x81727686, 0xc0a4f84a, 0xd48d6050, 0x9fe7e391, 0x415a9e7e, 0x9ef18e97, 0x62b0e5f3 },
		{ 0xec1f9ffc, 0xf594d74f, 0x7a205440, 0xd895fa9d, 0x001ae4e3, 0x117e2e55, 0x84c9f4ce, 0xa554c324 },
		{ 0x2872df5b, 0xef7c8905, 0x286efebd, 0x2ed349ee, 0xe27ff578, 0x85937e44, 0xb2c4a50f, 0x7f5928eb },
		// Round 14 (Function0)
		{ 0x37695f70, 0x04771bc7, 0x4a3124b3, 0xe720b951, 0xf128865e, 0xe843fe74, 0x65e4d61d, 0x8a87d423 },
		{ 0xa3e8297d, 0xfb301b1d, 0xf2947692, 0xe01bdc5b, 0x097acbdd, 0x4f4924da, 0xc1d9309b, 0xbf829cf2 },
		{ 0x31bae7a4, 0x32fcae3b, 0xffbf70b4, 0x39d3bb53, 0x0544320d, 0xc1c39f45, 0x48bcf8de, 0xa08b29e0 },
		{ 0xfd05c9e5, 0x01b771a2, 0x0f09aef7, 0x95ed44e3, 0x12347094, 0x368e3be9, 0x34f19042, 0x4a982f4f },
		{ 0x631d4088, 0xf14abb7e, 0x15f66ca0, 0x30c60ae2, 0x4b44c147, 0xc5b67046, 0xffaf5287, 0xe68c6ecc },
		{ 0x56a4d5a4, 0x45ce5773, 0x00ca4fbd, 0xadd16430, 0x4b849dda, 0x68cea6e8, 0xae183ec8, 0x67255c14 },
		{ 0xf28cdaa3, 0x20b2601f, 0x16e10ecb, 0x7b846fc2, 0x5806e933, 0x7facced1, 0x9a99949a, 0x1885d1a0 },
		// Round 21 (Function0)
		{ 0xa15b5932, 0x67633d9f, 0xd319dd8d, 0xba6b04e4, 0xc01c9a50, 0xab19caf6, 0x46b4a5aa, 0x7eee560b },
		{ 0xea79b11f, 0x5aac571d, 0x742128a9, 0x76d35075, 0x35f7bde9, 0xfec2463a, 0xee51363b, 0x01707da3 },
		{ 0xafc135f7, 0x15638341, 0x42d8a498, 0xa8db3aea, 0x20eced78, 0x4d3bc3fa, 0x79676b9e, 0x832c8332 },
		{ 0x1f3b40a7, 0x6c4e3ee7, 0xf347271c, 0xfd4f21d2, 0x34f04059, 0x398dfdb8, 0x9a762db7, 0xef5957dc },
		{ 0x490c9b8d, 0xd0ae3b7d, 0xdaeb492b, 0x84558d7a, 0x49d7a25b, 0xf0e9a5f5, 0x0d70f368, 0x658ef8e4 },
		{ 0xf4a2b8a0, 0x92946891, 0x533b1036, 0x4f88e856, 0x9e07a80c, 0x555cb05b, 0x5aec3e75, 0x4cbcbaf8 },
		{ 0x993bbbe3, 0x28acae64, 0x7b9487f3, 0x6db334dc, 0xd6f4da75, 0x50a5346c, 0x5d1c6b72, 0x71db28b8 },
		// Round 28 (Function0)
		{ 0xf2e261f8, 0xf1bcac1c, 0x2a518d10, 0xa23fce43, 0x3364dbe3, 0x3cd1bb67, 0xfc75dd59, 0xb043e802 },
		{ 0xca5b0a33, 0xc3943b92, 0x75a12988, 0x1e4d790e, 0x4d19347f, 0xd7757479, 0x5c5316b4, 0x3fafeeb6 },
		{ 0xf7d4a8ea, 0x5324a326, 0x21391abe, 0xd23c32ba, 0x097ef45c, 0x4a17a344, 0x5127234c, 0xadd5a66d },
		{ 0xa63e1db5, 0xa17cf84c, 0x08c9f2af, 0x4d608672, 0x983d5983, 0xcc3ee246, 0x563c6b91, 0xf6c76e08 },
		{ 0xb333982f, 0xe8b6f406, 0x5e76bcb1, 0x36d4c1be, 0xa566d62b, 0x1582ee74, 0x2ae6c4ef, 0x6321efbc },
		{ 0x0d4ec1fd, 0x1614c17e, 0x69c953f4, 0x16fae006, 0xc45a7da7, 0x3daf907e, 0x26585806, 0x3f9d6328 },
		{ 0xe3f2c9d2, 0x16512a74, 0x0cd29b00, 0x9832e0f2, 0x30ceaa5f, 0xd830eb0d, 0x300cd4b7, 0x9af8cee3 },
		// Round 35 (Function0)
		{ 0x7b9ec54b, 0x574d239b, 0x9279f1b5, 0x316796e6, 0x6ee651ff, 0xf3a6e6cc, 0xd3688604, 0x05750a17 },
		{ 0xd98176b1, 0xb3cb2bf4, 0xce6c3213, 0x47154778, 0x8452173c, 0x825446ff, 0x62a205f8, 0x486a9323 },
		{ 0x0758df38, 0x442e7031, 0x65655e4e, 0x86ca0bd0, 0x897cfcf2, 0xa20940f0, 0x8e5086fc, 0x4e477830 },
		{ 0x39eea065, 0x26b29721, 0x8338f7d1, 0x6ff81301, 0x37e95ef7, 0xd1ed44a3, 0xbd3a2ce4, 0xe7de9fef },
		{ 0x15dfa08b, 0x7ceca7d8, 0xd9922576, 0x7eb027ab, 0xf6f7853c, 0xda7d8d53, 0xbe42dc12, 0xdea83eaa },
		{ 0x93ce25aa, 0xdaef5fc0, 0xd86902bd, 0xa5194a17, 0xfd43f65a, 0x33664d97, 0xf908731a, 0x6a21fd4c },
		{ 0x3198b435, 0xa163d09a, 0x701541db, 0x72409751, 0xbb0f1eea, 0xbf9d75f6, 0x9b54cded, 0xe26f4791 }
		// 42 rounds...
};

__device__ __forceinline__
static void SWAP4(uint32_t *x) {
#pragma nounroll
	// y is used as tmp register too
	for (uint32_t y = 0; y<4; y++, ++x) {
		asm("and.b32 %1, %0, 0xF0F0F0F0;"
			"xor.b32 %0, %0, %1;"
			"shr.b32 %1, %1, 4;"
			"vshl.u32.u32.u32.clamp.add %0, %0, 4, %1;\n\t"
			: "+r"(*x) : "r"(y));
	}
}

__device__ __forceinline__
static void SWAP2(uint32_t *x) {
#pragma nounroll
	// y is used as tmp register too
	for (uint32_t y = 0; y<4; y++, ++x) {
		asm("and.b32 %1, %0, 0xCCCCCCCC;"
			"xor.b32 %0, %0, %1;"
			"shr.b32 %1, %1, 2;"
			"vshl.u32.u32.u32.clamp.add %0, %0, 2, %1;\n\t"
			: "+r"(*x) : "r"(y));
	}
}

__device__ __forceinline__
static void SWAP1(uint32_t *x) {
#pragma nounroll
	// y is used as tmp register too
	for (uint32_t y = 0; y<4; y++, ++x) {
		asm("and.b32 %1, %0, 0xAAAAAAAA;"
			"xor.b32 %0, %0, %1;"
			"shr.b32 %1, %1, 1;"
			"vshl.u32.u32.u32.clamp.add %0, %0, 1, %1;\n\t"
			: "+r"(*x) : "r"(y));
	}
}

/*swapping bits 16i||16i+1||......||16i+7  with bits 16i+8||16i+9||......||16i+15 of 32-bit x*/
//#define SWAP8(x)   (x) = ((((x) & 0x00ff00ffUL) << 8) | (((x) & 0xff00ff00UL) >> 8));
#define SWAP8(x) (x) = __byte_perm(x, x, 0x2301);
/*swapping bits 32i||32i+1||......||32i+15 with bits 32i+16||32i+17||......||32i+31 of 32-bit x*/
//#define SWAP16(x)  (x) = ((((x) & 0x0000ffffUL) << 16) | (((x) & 0xffff0000UL) >> 16));
#define SWAP16(x) (x) = __byte_perm(x, x, 0x1032);

/*The MDS transform*/
#define L(m0,m1,m2,m3,m4,m5,m6,m7) \
      (m4) ^= (m1);                \
      (m5) ^= (m2);                \
      (m6) ^= (m0) ^ (m3);         \
      (m7) ^= (m0);                \
      (m0) ^= (m5);                \
      (m1) ^= (m6);                \
      (m2) ^= (m4) ^ (m7);         \
      (m3) ^= (m4);

/*The Sbox*/
#define Sbox(m0,m1,m2,m3,cc)       \
      m3  = ~(m3);                 \
      m0 ^= ((~(m2)) & (cc));      \
      temp0 = (cc) ^ ((m0) & (m1));\
      m0 ^= ((m2) & (m3));         \
      m3 ^= ((~(m1)) & (m2));      \
      m1 ^= ((m0) & (m2));         \
      m2 ^= ((m0) & (~(m3)));      \
      m0 ^= ((m1) | (m3));         \
      m3 ^= ((m1) & (m2));         \
      m1 ^= (temp0 & (m0));        \
      m2 ^= temp0;

__device__ __forceinline__
static void Sbox_and_MDS_layer(uint32_t x[8][4], const int rnd)
{
	uint2* cc = (uint2*)&c_E8_bslice32[rnd];

	// Sbox and MDS layer
#pragma unroll
	for (int i = 0; i < 4; i++, ++cc) {
		uint32_t temp0;
		Sbox(x[0][i], x[2][i], x[4][i], x[6][i], cc->x);
		Sbox(x[1][i], x[3][i], x[5][i], x[7][i], cc->y);
		L(x[0][i], x[2][i], x[4][i], x[6][i], x[1][i], x[3][i], x[5][i], x[7][i]);
	}
}

static __device__ __forceinline__ void RoundFunction0(uint32_t x[8][4], uint32_t roundnumber)
{
	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
		SWAP1(x[j]);
	}
}

static __device__ __forceinline__ void RoundFunction1(uint32_t x[8][4], uint32_t roundnumber)
{
	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
		SWAP2(x[j]);
	}
}

static __device__ __forceinline__ void RoundFunction2(uint32_t x[8][4], uint32_t roundnumber)
{
	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
		SWAP4(x[j]);
	}
}

static __device__ __forceinline__ void RoundFunction3(uint32_t x[8][4], uint32_t roundnumber)
{
	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP8(x[j][i]);
	}
}

static __device__ __forceinline__ void RoundFunction4(uint32_t x[8][4], uint32_t roundnumber)
{
	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP16(x[j][i]);
	}
}

static __device__ __forceinline__ void RoundFunction5(uint32_t x[8][4], uint32_t roundnumber)
{
	uint32_t temp0;

	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 2
		for (int i = 0; i < 4; i = i+2) {
			temp0 = x[j][i]; x[j][i] = x[j][i+1]; x[j][i+1] = temp0;
		}
	}
}

static __device__ __forceinline__ void RoundFunction6(uint32_t x[8][4], uint32_t roundnumber)
{
	uint32_t temp0;

	Sbox_and_MDS_layer(x, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 2
		for (int i = 0; i < 2; i++) {
			temp0 = x[j][i]; x[j][i] = x[j][i+2]; x[j][i+2] = temp0;
		}
	}
}

/*The bijective function E8, in bitslice form */
static __device__ __forceinline__ void E8(uint32_t x[8][4])
{
    /*perform 6 rounds*/
#pragma unroll 1
    for (int i = 0; i < 42; i+=7)
	{
		RoundFunction0(x, i);
		RoundFunction1(x, i + 1);
		RoundFunction2(x, i + 2);
		RoundFunction3(x, i + 3);
		RoundFunction4(x, i + 4);
		RoundFunction5(x, i + 5);
		RoundFunction6(x, i + 6);
	}
}

#define U32TO64_LE(p) \
    (((uint64_t)(*p)) | (((uint64_t)(*(p + 1))) << 32))

#define U64TO32_LE(p, v) \
    *p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);
 
__constant__ uint2 c_keccak_round_constants[24] = {
		{ 0x00000001ul, 0x00000000 }, { 0x00008082ul, 0x00000000 },
		{ 0x0000808aul, 0x80000000 }, { 0x80008000ul, 0x80000000 },
		{ 0x0000808bul, 0x00000000 }, { 0x80000001ul, 0x00000000 },
		{ 0x80008081ul, 0x80000000 }, { 0x00008009ul, 0x80000000 },
		{ 0x0000008aul, 0x00000000 }, { 0x00000088ul, 0x00000000 },
		{ 0x80008009ul, 0x00000000 }, { 0x8000000aul, 0x00000000 },
		{ 0x8000808bul, 0x00000000 }, { 0x0000008bul, 0x80000000 },
		{ 0x00008089ul, 0x80000000 }, { 0x00008003ul, 0x80000000 },
		{ 0x00008002ul, 0x80000000 }, { 0x00000080ul, 0x80000000 },
		{ 0x0000800aul, 0x00000000 }, { 0x8000000aul, 0x80000000 },
		{ 0x80008081ul, 0x80000000 }, { 0x00008080ul, 0x80000000 },
		{ 0x80000001ul, 0x00000000 }, { 0x80008008ul, 0x80000000 }
};

#define bitselect(a, b, c) ((a) ^ ((c) & ((b) ^ (a))))

__global__ __launch_bounds__(256,3)
void quark_jh512Keccak512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        const uint32_t nounce =  (startNounce + thread);

        const int hashPosition = nounce - startNounce;
        uint32_t *Hash = &g_hash[16 * hashPosition]; 
		uint32_t x[8][4] = {
				{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a },
				{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2 },
				{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea },
				{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba },
				{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e },
				{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d },
				{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657 },
				{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc } };

#pragma unroll 16
		for (int i = 0; i < 16; i++)  x[i >> 2][i & 3] ^= ((uint32_t*)Hash)[i];
		E8(x);
#pragma unroll 16
		for (int i = 0; i < 16; i++) x[(16 + i) >> 2][(16 + i) & 3] ^= ((uint32_t*)Hash)[i];

		x[0 >> 2][0 & 3] ^= 0x80;
		x[15 >> 2][15 & 3] ^= 0x00020000;
		E8(x);
		x[(16 + 0) >> 2][(16 + 0) & 3] ^= 0x80;
		x[(16 + 15) >> 2][(16 + 15) & 3] ^= 0x00020000;

		uint2 s[25] =
		{
			{x[4][0], x[4][1]}, {x[4][2], x[4][3]}, {x[5][0], x[5][1]}, {x[5][2], x[5][3]},
			{x[6][0], x[6][1]}, {x[6][2], x[6][3]}, {x[7][0], x[7][1]}, {x[7][2], x[7][3]},
			{1, 0x80000000}, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 },
			{ 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 },
			{ 0, 0 }
		};
		uint2 bc[5], tmpxor[5], tmp1, tmp2;

		tmpxor[0] = s[0] ^ s[5];
		tmpxor[1] = s[1] ^ s[6];
		tmpxor[2] = s[2] ^ s[7];
		tmpxor[3] = s[3] ^ s[8];
		tmpxor[4] = s[4];

		bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
		bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
		bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
		bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
		bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

		tmp1 = s[1] ^ bc[0];

		s[0] = s[0] ^ bc[4];
		s[1] = ROL2(s[6] ^ bc[0], 44);
		s[6] = ROL2(bc[3], 20);
		s[9] = ROL2(bc[1], 61);
		s[22] = ROL2(bc[3], 39);
		s[14] = ROL2(bc[4], 18);
		s[20] = ROL2(s[2] ^ bc[1], 62);
		s[2] = ROL2(bc[1], 43);
		s[12] = ROL2(bc[2], 25);
		s[13] = ROL8(bc[3]);
		s[19] = ROR8(bc[2]);
		s[23] = ROL2(bc[4], 41);
		s[15] = ROL2(s[4] ^ bc[3], 27);
		s[4] = ROL2(bc[3], 14);
		s[24] = ROL2(bc[0], 2);
		s[21] = ROL2(s[8] ^ bc[2], 55);
		s[8] = ROL2(bc[0], 45);
		s[16] = ROL2(s[5] ^ bc[4], 36);
		s[5] = ROL2(s[3] ^ bc[2], 28);
		s[3] = ROL2(bc[2], 21);
		s[18] = ROL2(bc[1], 15);
		s[17] = ROL2(bc[0], 10);
		s[11] = ROL2(s[7] ^ bc[1], 6);
		s[7] = ROL2(bc[4], 3);
		s[10] = ROL2(tmp1, 1);

		tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
		tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
		tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
		tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
		tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
		s[0].x ^= 1;
#pragma unroll 1
		for (int i = 1; i < 24; ++i)
		{

#pragma unroll
			for (uint32_t x = 0; x < 5; x++)
				tmpxor[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];

			bc[0] = tmpxor[0] ^ ROL2(tmpxor[2], 1);
			bc[1] = tmpxor[1] ^ ROL2(tmpxor[3], 1);
			bc[2] = tmpxor[2] ^ ROL2(tmpxor[4], 1);
			bc[3] = tmpxor[3] ^ ROL2(tmpxor[0], 1);
			bc[4] = tmpxor[4] ^ ROL2(tmpxor[1], 1);

			tmp1 = s[1] ^ bc[0];

			s[0] ^= bc[4];
			s[1] = ROL2(s[6] ^ bc[0], 44);
			s[6] = ROL2(s[9] ^ bc[3], 20);
			s[9] = ROL2(s[22] ^ bc[1], 61);
			s[22] = ROL2(s[14] ^ bc[3], 39);
			s[14] = ROL2(s[20] ^ bc[4], 18);
			s[20] = ROL2(s[2] ^ bc[1], 62);
			s[2] = ROL2(s[12] ^ bc[1], 43);
			s[12] = ROL2(s[13] ^ bc[2], 25);
			s[13] = ROL8(s[19] ^ bc[3]);
			s[19] = ROR8(s[23] ^ bc[2]);
			s[23] = ROL2(s[15] ^ bc[4], 41);
			s[15] = ROL2(s[4] ^ bc[3], 27);
			s[4] = ROL2(s[24] ^ bc[3], 14);
			s[24] = ROL2(s[21] ^ bc[0], 2);
			s[21] = ROL2(s[8] ^ bc[2], 55);
			s[8] = ROL2(s[16] ^ bc[0], 45);
			s[16] = ROL2(s[5] ^ bc[4], 36);
			s[5] = ROL2(s[3] ^ bc[2], 28);
			s[3] = ROL2(s[18] ^ bc[2], 21);
			s[18] = ROL2(s[17] ^ bc[1], 15);
			s[17] = ROL2(s[11] ^ bc[0], 10);
			s[11] = ROL2(s[7] ^ bc[1], 6);
			s[7] = ROL2(s[10] ^ bc[4], 3);
			s[10] = ROL2(tmp1, 1);

			tmp1 = s[0]; tmp2 = s[1]; s[0] = bitselect(s[0] ^ s[2], s[0], s[1]); s[1] = bitselect(s[1] ^ s[3], s[1], s[2]); s[2] = bitselect(s[2] ^ s[4], s[2], s[3]); s[3] = bitselect(s[3] ^ tmp1, s[3], s[4]); s[4] = bitselect(s[4] ^ tmp2, s[4], tmp1);
			tmp1 = s[5]; tmp2 = s[6]; s[5] = bitselect(s[5] ^ s[7], s[5], s[6]); s[6] = bitselect(s[6] ^ s[8], s[6], s[7]); s[7] = bitselect(s[7] ^ s[9], s[7], s[8]); s[8] = bitselect(s[8] ^ tmp1, s[8], s[9]); s[9] = bitselect(s[9] ^ tmp2, s[9], tmp1);
			tmp1 = s[10]; tmp2 = s[11]; s[10] = bitselect(s[10] ^ s[12], s[10], s[11]); s[11] = bitselect(s[11] ^ s[13], s[11], s[12]); s[12] = bitselect(s[12] ^ s[14], s[12], s[13]); s[13] = bitselect(s[13] ^ tmp1, s[13], s[14]); s[14] = bitselect(s[14] ^ tmp2, s[14], tmp1);
			tmp1 = s[15]; tmp2 = s[16]; s[15] = bitselect(s[15] ^ s[17], s[15], s[16]); s[16] = bitselect(s[16] ^ s[18], s[16], s[17]); s[17] = bitselect(s[17] ^ s[19], s[17], s[18]); s[18] = bitselect(s[18] ^ tmp1, s[18], s[19]); s[19] = bitselect(s[19] ^ tmp2, s[19], tmp1);
			tmp1 = s[20]; tmp2 = s[21]; s[20] = bitselect(s[20] ^ s[22], s[20], s[21]); s[21] = bitselect(s[21] ^ s[23], s[21], s[22]); s[22] = bitselect(s[22] ^ s[24], s[22], s[23]); s[23] = bitselect(s[23] ^ tmp1, s[23], s[24]); s[24] = bitselect(s[24] ^ tmp2, s[24], tmp1);
			s[0] ^= c_keccak_round_constants[i];
		}

		uint2 *outputhash = (uint2 *)Hash;
#pragma unroll 16
		for (int i = 0; i<8; i++)
			outputhash[i] = s[i];
	}
}


__host__ void cuda_jh512Keccak512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
    const uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

	quark_jh512Keccak512_gpu_hash_64 << <grid, block>> >(threads, startNounce, d_hash);
}

