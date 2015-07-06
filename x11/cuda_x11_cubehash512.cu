#include "cuda_helper.h"

#define ROUND_EVEN   \
		xg = (x0 + xg); \
		x0 = ROTL32(x0, 7); \
		xh = (x1 + xh); \
		x1 = ROTL32(x1, 7); \
		xi = (x2 + xi); \
		x2 = ROTL32(x2, 7); \
		xj = (x3 + xj); \
		x3 = ROTL32(x3, 7); \
		xk = (x4 + xk); \
		x4 = ROTL32(x4, 7); \
		xl = (x5 + xl); \
		x5 = ROTL32(x5, 7); \
		xm = (x6 + xm); \
		x6 = ROTL32(x6, 7); \
		xn = (x7 + xn); \
		x7 = ROTL32(x7, 7); \
		xo = (x8 + xo); \
		x8 = ROTL32(x8, 7); \
		xp = (x9 + xp); \
		x9 = ROTL32(x9, 7); \
		xq = (xa + xq); \
		xa = ROTL32(xa, 7); \
		xr = (xb + xr); \
		xb = ROTL32(xb, 7); \
		xs = (xc + xs); \
		xc = ROTL32(xc, 7); \
		xt = (xd + xt); \
		xd = ROTL32(xd, 7); \
		xu = (xe + xu); \
		xe = ROTL32(xe, 7); \
		xv = (xf + xv); \
		xf = ROTL32(xf, 7); \
		x8 ^= xg; \
		x9 ^= xh; \
		xa ^= xi; \
		xb ^= xj; \
		xc ^= xk; \
		xd ^= xl; \
		xe ^= xm; \
		xf ^= xn; \
		x0 ^= xo; \
		x1 ^= xp; \
		x2 ^= xq; \
		x3 ^= xr; \
		x4 ^= xs; \
		x5 ^= xt; \
		x6 ^= xu; \
		x7 ^= xv; \
		xi = (x8 + xi); \
		x8 = ROTL32(x8, 11); \
		xj = (x9 + xj); \
		x9 = ROTL32(x9, 11); \
		xg = (xa + xg); \
		xa = ROTL32(xa, 11); \
		xh = (xb + xh); \
		xb = ROTL32(xb, 11); \
		xm = (xc + xm); \
		xc = ROTL32(xc, 11); \
		xn = (xd + xn); \
		xd = ROTL32(xd, 11); \
		xk = (xe + xk); \
		xe = ROTL32(xe, 11); \
		xl = (xf + xl); \
		xf = ROTL32(xf, 11); \
		xq = (x0 + xq); \
		x0 = ROTL32(x0, 11); \
		xr = (x1 + xr); \
		x1 = ROTL32(x1, 11); \
		xo = (x2 + xo); \
		x2 = ROTL32(x2, 11); \
		xp = (x3 + xp); \
		x3 = ROTL32(x3, 11); \
		xu = (x4 + xu); \
		x4 = ROTL32(x4, 11); \
		xv = (x5 + xv); \
		x5 = ROTL32(x5, 11); \
		xs = (x6 + xs); \
		x6 = ROTL32(x6, 11); \
		xt = (x7 + xt); \
		x7 = ROTL32(x7, 11); \
		xc ^= xi; \
		xd ^= xj; \
		xe ^= xg; \
		xf ^= xh; \
		x8 ^= xm; \
		x9 ^= xn; \
		xa ^= xk; \
		xb ^= xl; \
		x4 ^= xq; \
		x5 ^= xr; \
		x6 ^= xo; \
		x7 ^= xp; \
		x0 ^= xu; \
		x1 ^= xv; \
		x2 ^= xs; \
		x3 ^= xt; 

#define ROUND_ODD    \
		xj = (xc + xj); \
		xc = ROTL32(xc, 7); \
		xi = (xd + xi); \
		xd = ROTL32(xd, 7); \
		xh = (xe + xh); \
		xe = ROTL32(xe, 7); \
		xg = (xf + xg); \
		xf = ROTL32(xf, 7); \
		xn = (x8 + xn); \
		x8 = ROTL32(x8, 7); \
		xm = (x9 + xm); \
		x9 = ROTL32(x9, 7); \
		xl = (xa + xl); \
		xa = ROTL32(xa, 7); \
		xk = (xb + xk); \
		xb = ROTL32(xb, 7); \
		xr = (x4 + xr); \
		x4 = ROTL32(x4, 7); \
		xq = (x5 + xq); \
		x5 = ROTL32(x5, 7); \
		xp = (x6 + xp); \
		x6 = ROTL32(x6, 7); \
		xo = (x7 + xo); \
		x7 = ROTL32(x7, 7); \
		xv = (x0 + xv); \
		x0 = ROTL32(x0, 7); \
		xu = (x1 + xu); \
		x1 = ROTL32(x1, 7); \
		xt = (x2 + xt); \
		x2 = ROTL32(x2, 7); \
		xs = (x3 + xs); \
		x3 = ROTL32(x3, 7); \
		x4 ^= xj; \
		x5 ^= xi; \
		x6 ^= xh; \
		x7 ^= xg; \
		x0 ^= xn; \
		x1 ^= xm; \
		x2 ^= xl; \
		x3 ^= xk; \
		xc ^= xr; \
		xd ^= xq; \
		xe ^= xp; \
		xf ^= xo; \
		x8 ^= xv; \
		x9 ^= xu; \
		xa ^= xt; \
		xb ^= xs; \
		xh = (x4 + xh); \
		x4 = ROTL32(x4, 11); \
		xg = (x5 + xg); \
		x5 = ROTL32(x5, 11); \
		xj = (x6 + xj); \
		x6 = ROTL32(x6, 11); \
		xi = (x7 + xi); \
		x7 = ROTL32(x7, 11); \
		xl = (x0 + xl); \
		x0 = ROTL32(x0, 11); \
		xk = (x1 + xk); \
		x1 = ROTL32(x1, 11); \
		xn = (x2 + xn); \
		x2 = ROTL32(x2, 11); \
		xm = (x3 + xm); \
		x3 = ROTL32(x3, 11); \
		xp = (xc + xp); \
		xc = ROTL32(xc, 11); \
		xo = (xd + xo); \
		xd = ROTL32(xd, 11); \
		xr = (xe + xr); \
		xe = ROTL32(xe, 11); \
		xq = (xf + xq); \
		xf = ROTL32(xf, 11); \
		xt = (x8 + xt); \
		x8 = ROTL32(x8, 11); \
		xs = (x9 + xs); \
		x9 = ROTL32(x9, 11); \
		xv = (xa + xv); \
		xa = ROTL32(xa, 11); \
		xu = (xb + xu); \
		xb = ROTL32(xb, 11); \
		x0 ^= xh; \
		x1 ^= xg; \
		x2 ^= xj; \
		x3 ^= xi; \
		x4 ^= xl; \
		x5 ^= xk; \
		x6 ^= xn; \
		x7 ^= xm; \
		x8 ^= xp; \
		x9 ^= xo; \
		xa ^= xr; \
		xb ^= xq; \
		xc ^= xt; \
		xd ^= xs; \
		xe ^= xv; \
		xf ^= xu; 

#define SIXTEEN_ROUNDS \
		for (int j = 0; j < 8; j ++) { \
			ROUND_EVEN; \
			ROUND_ODD;}
__global__	
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (startNounce + thread);
        int hashPosition = nounce - startNounce;
        uint32_t *Hash = &g_hash[16 * hashPosition];

		uint32_t x0 = (0x2AEA2A61), x1 = (0x50F494D4), x2 = (0x2D538B8B), x3 = (0x4167D83E);
		uint32_t x4 = (0x3FEE2313), x5 = (0xC701CF8C), x6 = (0xCC39968E), x7 = (0x50AC5695);
		uint32_t x8 = (0x4D42C787), x9 = (0xA647A8B3), xa = (0x97CF0BEF), xb = (0x825B4537);
		uint32_t xc = (0xEEF864D2), xd = (0xF22090C4), xe = (0xD0E5CD33), xf = (0xA23911AE);
		uint32_t xg = (0xFCD398D9), xh = (0x148FE485), xi = (0x1B017BEF), xj = (0xB6444532);
		uint32_t xk = (0x6A536159), xl = (0x2FF5781C), xm = (0x91FA7934), xn = (0x0DBADEA9);
		uint32_t xo = (0xD65C8A2B), xp = (0xA5A70E75), xq = (0xB1C62456), xr = (0xBC796576);
		uint32_t xs = (0x1921C8F7), xt = (0xE7989AF1), xu = (0x7795D246), xv = (0xD43E3B44);

		x0 ^= Hash[0];
		x1 ^= Hash[1];
		x2 ^= Hash[2];
		x3 ^= Hash[3];
		x4 ^= Hash[4];
		x5 ^= Hash[5];
		x6 ^= Hash[6];
		x7 ^= Hash[7];

		SIXTEEN_ROUNDS;

		x0 ^= (Hash[8]);
		x1 ^= (Hash[9]);
		x2 ^= (Hash[10]);
		x3 ^= (Hash[11]);
		x4 ^= (Hash[12]);
		x5 ^= (Hash[13]);
		x6 ^= (Hash[14]);
		x7 ^= (Hash[15]);

		SIXTEEN_ROUNDS;
		x0 ^= 0x80;

		SIXTEEN_ROUNDS;
		xv ^= 1;

		for (int i = 3; i < 13; i++) 
		{
			SIXTEEN_ROUNDS;
		}

		Hash[0] = x0;
		Hash[1] = x1;
		Hash[2] = x2;
		Hash[3] = x3;
		Hash[4] = x4;
		Hash[5] = x5;
		Hash[6] = x6;
		Hash[7] = x7;
		Hash[8] = x8;
		Hash[9] = x9;
		Hash[10] = xa;
		Hash[11] = xb;
		Hash[12] = xc;
		Hash[13] = xd;
		Hash[14] = xe;
		Hash[15] = xf;
    }
}
__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash);
}

