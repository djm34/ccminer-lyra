/*
 * Shabal-512 for X14/X15 (STUB)
 */
#include "cuda_helper.h"


/* $Id: shabal.c 175 2010-05-07 16:03:20Z tp $ */
/*
 * Shabal implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010 Projet RNRT SAPHIR
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
 * @author Thomas Pornin <thomas.pornin@cryptolog.com>
 */

/*
 * Part of this code was automatically generated (the part between
 * the "BEGIN" and "END" markers).
 */

#define sM    16

#define O1   13
#define O2    9
#define O3    6

/*
 * We copy the state into local variables, so that the compiler knows
 * that it can optimize them at will.
 */

/* BEGIN -- automatically generated code. */

#define INPUT_BLOCK_ADD  \
		B0 = B0 + M0; \
		B1 = B1 + M1; \
		B2 = B2 + M2; \
		B3 = B3 + M3; \
		B4 = B4 + M4; \
		B5 = B5 + M5; \
		B6 = B6 + M6; \
		B7 = B7 + M7; \
		B8 = B8 + M8; \
		B9 = B9 + M9; \
		BA = BA + MA; \
		BB = BB + MB; \
		BC = BC + MC; \
		BD = BD + MD; \
		BE = BE + ME; \
		BF = BF + MF; \

#define INPUT_BLOCK_SUB \
		C0 = C0 - M0; \
		C1 = C1 - M1; \
		C2 = C2 - M2; \
		C3 = C3 - M3; \
		C4 = C4 - M4; \
		C5 = C5 - M5; \
		C6 = C6 - M6; \
		C7 = C7 - M7; \
		C8 = C8 - M8; \
		C9 = C9 - M9; \
		CA = CA - MA; \
		CB = CB - MB; \
		CC = CC - MC; \
		CD = CD - MD; \
		CE = CE - ME; \
		CF = CF - MF; \

#define XOR_W  \
		A00 ^= Wlow; \
		A01 ^= Whigh; \

#define SWAP(v1, v2) \
		tmp = (v1); \
		(v1) = (v2); \
		(v2) = tmp; \

#define SWAP_BC   \
		SWAP(B0, C0); \
		SWAP(B1, C1); \
		SWAP(B2, C2); \
		SWAP(B3, C3); \
		SWAP(B4, C4); \
		SWAP(B5, C5); \
		SWAP(B6, C6); \
		SWAP(B7, C7); \
		SWAP(B8, C8); \
		SWAP(B9, C9); \
		SWAP(BA, CA); \
		SWAP(BB, CB); \
		SWAP(BC, CC); \
		SWAP(BD, CD); \
		SWAP(BE, CE); \
		SWAP(BF, CF); \

#define PERM_ELT(xa0, xa1, xb0, xb1, xb2, xb3, xc, xm) \
		xa0 = ((xa0 \
			^ (ROTL32(xa1, 15) * 5U) \
			^ xc) * 3U) \
			^ xb1 ^ (xb2 & ~xb3) ^ xm; \
		xb0 = (~(ROTL32(xb0, 1) ^ xa0)); \

#define PERM_STEP_0 \
		PERM_ELT(A00, A0B, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A01, A00, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A02, A01, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A03, A02, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A04, A03, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A05, A04, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A06, A05, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A07, A06, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A08, A07, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A09, A08, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A0A, A09, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A0B, A0A, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A00, A0B, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A01, A00, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A02, A01, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A03, A02, BF, BC, B8, B5, C9, MF); \

#define PERM_STEP_1 \
		PERM_ELT(A04, A03, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A05, A04, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A06, A05, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A07, A06, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A08, A07, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A09, A08, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A0A, A09, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A0B, A0A, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A00, A0B, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A01, A00, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A02, A01, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A03, A02, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A04, A03, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A05, A04, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A06, A05, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A07, A06, BF, BC, B8, B5, C9, MF); \

#define PERM_STEP_2 \
		PERM_ELT(A08, A07, B0, BD, B9, B6, C8, M0); \
		PERM_ELT(A09, A08, B1, BE, BA, B7, C7, M1); \
		PERM_ELT(A0A, A09, B2, BF, BB, B8, C6, M2); \
		PERM_ELT(A0B, A0A, B3, B0, BC, B9, C5, M3); \
		PERM_ELT(A00, A0B, B4, B1, BD, BA, C4, M4); \
		PERM_ELT(A01, A00, B5, B2, BE, BB, C3, M5); \
		PERM_ELT(A02, A01, B6, B3, BF, BC, C2, M6); \
		PERM_ELT(A03, A02, B7, B4, B0, BD, C1, M7); \
		PERM_ELT(A04, A03, B8, B5, B1, BE, C0, M8); \
		PERM_ELT(A05, A04, B9, B6, B2, BF, CF, M9); \
		PERM_ELT(A06, A05, BA, B7, B3, B0, CE, MA); \
		PERM_ELT(A07, A06, BB, B8, B4, B1, CD, MB); \
		PERM_ELT(A08, A07, BC, B9, B5, B2, CC, MC); \
		PERM_ELT(A09, A08, BD, BA, B6, B3, CB, MD); \
		PERM_ELT(A0A, A09, BE, BB, B7, B4, CA, ME); \
		PERM_ELT(A0B, A0A, BF, BC, B8, B5, C9, MF); \

#define APPLY_P  \
		B0 = ROTL32(B0, 17); \
		B1 = ROTL32(B1, 17); \
		B2 = ROTL32(B2, 17); \
		B3 = ROTL32(B3, 17); \
		B4 = ROTL32(B4, 17); \
		B5 = ROTL32(B5, 17); \
		B6 = ROTL32(B6, 17); \
		B7 = ROTL32(B7, 17); \
		B8 = ROTL32(B8, 17); \
		B9 = ROTL32(B9, 17); \
		BA = ROTL32(BA, 17); \
		BB = ROTL32(BB, 17); \
		BC = ROTL32(BC, 17); \
		BD = ROTL32(BD, 17); \
		BE = ROTL32(BE, 17); \
		BF = ROTL32(BF, 17); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \
		A0B = (A0B + C6); \
		A0A = (A0A + C5); \
		A09 = (A09 + C4); \
		A08 = (A08 + C3); \
		A07 = (A07 + C2); \
		A06 = (A06 + C1); \
		A05 = (A05 + C0); \
		A04 = (A04 + CF); \
		A03 = (A03 + CE); \
		A02 = (A02 + CD); \
		A01 = (A01 + CC); \
		A00 = (A00 + CB); \
		A0B = (A0B + CA); \
		A0A = (A0A + C9); \
		A09 = (A09 + C8); \
		A08 = (A08 + C7); \
		A07 = (A07 + C6); \
		A06 = (A06 + C5); \
		A05 = (A05 + C4); \
		A04 = (A04 + C3); \
		A03 = (A03 + C2); \
		A02 = (A02 + C1); \
		A01 = (A01 + C0); \
		A00 = (A00 + CF); \
		A0B = (A0B + CE); \
		A0A = (A0A + CD); \
		A09 = (A09 + CC); \
		A08 = (A08 + CB); \
		A07 = (A07 + CA); \
		A06 = (A06 + C9); \
		A05 = (A05 + C8); \
		A04 = (A04 + C7); \
		A03 = (A03 + C6); \
		A02 = (A02 + C5); \
		A01 = (A01 + C4); \
		A00 = (A00 + C3); \

#define APPLY_P_FINAL  \
		B0 = ROTL32(B0, 17); \
		B1 = ROTL32(B1, 17); \
		B2 = ROTL32(B2, 17); \
		B3 = ROTL32(B3, 17); \
		B4 = ROTL32(B4, 17); \
		B5 = ROTL32(B5, 17); \
		B6 = ROTL32(B6, 17); \
		B7 = ROTL32(B7, 17); \
		B8 = ROTL32(B8, 17); \
		B9 = ROTL32(B9, 17); \
		BA = ROTL32(BA, 17); \
		BB = ROTL32(BB, 17); \
		BC = ROTL32(BC, 17); \
		BD = ROTL32(BD, 17); \
		BE = ROTL32(BE, 17); \
		BF = ROTL32(BF, 17); \
		PERM_STEP_0; \
		PERM_STEP_1; \
		PERM_STEP_2; \

#define INCR_W if ((Wlow = (Wlow + 1)) == 0) \
			Whigh = (Whigh + 1); \
	


#if 0 /* other hash sizes init */

static const uint32_t A_init_192[] = {
	0xFD749ED4), 0xB798E530), 0x33904B6F), 0x46BDA85E),
	0x076934B4), 0x454B4058), 0x77F74527), 0xFB4CF465),
	0x62931DA9), 0xE778C8DB), 0x22B3998E), 0xAC15CFB9)
};

static const uint32_t B_init_192[] = {
	0x58BCBAC4), 0xEC47A08E), 0xAEE933B2), 0xDFCBC824),
	0xA7944804), 0xBF65BDB0), 0x5A9D4502), 0x59979AF7),
	0xC5CEA54E), 0x4B6B8150), 0x16E71909), 0x7D632319),
	0x930573A0), 0xF34C63D1), 0xCAF914B4), 0xFDD6612C)
};

static const uint32_t C_init_192[] = {
	0x61550878), 0x89EF2B75), 0xA1660C46), 0x7EF3855B),
	0x7297B58C), 0x1BC67793), 0x7FB1C723), 0xB66FC640),
	0x1A48B71C), 0xF0976D17), 0x088CE80A), 0xA454EDF3),
	0x1C096BF4), 0xAC76224B), 0x5215781C), 0xCD5D2669)
};

static const uint32_t A_init_224[] = {
	0xA5201467), 0xA9B8D94A), 0xD4CED997), 0x68379D7B),
	0xA7FC73BA), 0xF1A2546B), 0x606782BF), 0xE0BCFD0F),
	0x2F25374E), 0x069A149F), 0x5E2DFF25), 0xFAECF061)
};

static const uint32_t B_init_224[] = {
	0xEC9905D8), 0xF21850CF), 0xC0A746C8), 0x21DAD498),
	0x35156EEB), 0x088C97F2), 0x26303E40), 0x8A2D4FB5),
	0xFEEE44B6), 0x8A1E9573), 0x7B81111A), 0xCBC139F0),
	0xA3513861), 0x1D2C362E), 0x918C580E), 0xB58E1B9C)
};

static const uint32_t C_init_224[] = {
	0xE4B573A1), 0x4C1A0880), 0x1E907C51), 0x04807EFD),
	0x3AD8CDE5), 0x16B21302), 0x02512C53), 0x2204CB18),
	0x99405F2D), 0xE5B648A1), 0x70AB1D43), 0xA10C25C2),
	0x16F1AC05), 0x38BBEB56), 0x9B01DC60), 0xB1096D83)
};

static const uint32_t A_init_256[] = {
	0x52F84552), 0xE54B7999), 0x2D8EE3EC), 0xB9645191),
	0xE0078B86), 0xBB7C44C9), 0xD2B5C1CA), 0xB0D2EB8C),
	0x14CE5A45), 0x22AF50DC), 0xEFFDBC6B), 0xEB21B74A)
};

static const uint32_t B_init_256[] = {
	0xB555C6EE), 0x3E710596), 0xA72A652F), 0x9301515F),
	0xDA28C1FA), 0x696FD868), 0x9CB6BF72), 0x0AFE4002),
	0xA6E03615), 0x5138C1D4), 0xBE216306), 0xB38B8890),
	0x3EA8B96B), 0x3299ACE4), 0x30924DD4), 0x55CB34A5)
};

static const uint32_t C_init_256[] = {
	0xB405F031), 0xC4233EBA), 0xB3733979), 0xC0DD9D55),
	0xC51C28AE), 0xA327B8E1), 0x56C56167), 0xED614433),
	0x88B59D60), 0x60E2CEBA), 0x758B4B8B), 0x83E82A7F),
	0xBC968828), 0xE6E00BF7), 0xBA839E55), 0x9B491C60)
};

static const uint32_t A_init_384[] = {
	0xC8FCA331), 0xE55C504E), 0x003EBF26), 0xBB6B8D83),
	0x7B0448C1), 0x41B82789), 0x0A7C9601), 0x8D659CFF),
	0xB6E2673E), 0xCA54C77B), 0x1460FD7E), 0x3FCB8F2D)
};

static const uint32_t B_init_384[] = {
	0x527291FC), 0x2A16455F), 0x78E627E5), 0x944F169F),
	0x1CA6F016), 0xA854EA25), 0x8DB98ABE), 0xF2C62641),
	0x30117DCB), 0xCF5C4309), 0x93711A25), 0xF9F671B8),
	0xB01D2116), 0x333F4B89), 0xB285D165), 0x86829B36)
};

static const uint32_t C_init_384[] = {
	0xF764B11A), 0x76172146), 0xCEF6934D), 0xC6D28399),
	0xFE095F61), 0x5E6018B4), 0x5048ECF5), 0x51353261),
	0x6E6E36DC), 0x63130DAD), 0xA9C69BD6), 0x1E90EA0C),
	0x7C35073B), 0x28D95E6D), 0xAA340E0D), 0xCB3DEE70)
};
#endif


/***************************************************/
// GPU Hash Function
__global__ __launch_bounds__(256, 4)
void x14_shabal512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	const uint32_t d_A512[] = {
		0x20728DFD, 0x46C0BD53, 0xE782B699,0x55304632,
		0x71B4EF90, 0x0EA9E82C, 0xDBB930F1, 0xFAD06B8B,
		0xBE0CAE40, 0x8BD14410, 0x76D2ADAC, 0x28ACAB7F
	};

	const uint32_t d_B512[] = {
		0xC1099CB7, 0x07B385F3, 0xE7442C26, 0xCC8AD640,
		0xEB6F56C7, 0x1EA81AA9, 0x73B9D314, 0x1DE85D08,
		0x48910A5A, 0x893B22DB, 0xC5A0DF44, 0xBBC4324E,
		0x72D2F240, 0x75941D99, 0x6D8BDE82, 0xA1A7502B
	};

	const uint32_t d_C512[] = {
		0xD9BF68D1, 0x58BAD750, 0x56028CB2, 0x8134F359,
		0xB5D469D8, 0x941A8CC2, 0x418B2A6E, 0x04052780,
		0x7F07D787, 0x5194358F, 0x3C60D665, 0xBE97D79A,
		0x950C3434, 0xAED9A06D, 0x2537DC8D, 0x7CDB5969
	};

	if (thread < threads)
	{
		uint32_t nounce =  (startNounce + thread);
		uint32_t hashPosition = nounce - startNounce;
		uint32_t *Hash = &g_hash[hashPosition*16]; // [hashPosition * 8]
		uint32_t tmp;
		uint32_t A00 = d_A512[0], A01 = d_A512[1], A02 = d_A512[2], A03 = d_A512[3],
			A04 = d_A512[4], A05 = d_A512[5], A06 = d_A512[6], A07 = d_A512[7],
			A08 = d_A512[8], A09 = d_A512[9], A0A = d_A512[10], A0B = d_A512[11];
		uint32_t B0 = d_B512[0], B1 = d_B512[1], B2 = d_B512[2], B3 = d_B512[3],
			B4 = d_B512[4], B5 = d_B512[5], B6 = d_B512[6], B7 = d_B512[7],
			B8 = d_B512[8], B9 = d_B512[9], BA = d_B512[10], BB = d_B512[11],
			BC = d_B512[12], BD = d_B512[13], BE = d_B512[14], BF = d_B512[15];
		uint32_t C0 = d_C512[0], C1 = d_C512[1], C2 = d_C512[2], C3 = d_C512[3],
			C4 = d_C512[4], C5 = d_C512[5], C6 = d_C512[6], C7 = d_C512[7],
			C8 = d_C512[8], C9 = d_C512[9], CA = d_C512[10], CB = d_C512[11],
			CC = d_C512[12], CD = d_C512[13], CE = d_C512[14], CF = d_C512[15];
		uint32_t M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, MA, MB, MC, MD, ME, MF;

		M0 = Hash[0];
		M1 = Hash[1];
		M2 = Hash[2];
		M3 = Hash[3];
		M4 = Hash[4];
		M5 = Hash[5];
		M6 = Hash[6];
		M7 = Hash[7];

		M8 = Hash[8];
		M9 = Hash[9];
		MA = Hash[10];
		MB = Hash[11];
		MC = Hash[12];
		MD = Hash[13];
		ME = Hash[14];
		MF = Hash[15];

		INPUT_BLOCK_ADD;
		A00 ^= 1;
		APPLY_P;
		INPUT_BLOCK_SUB;
		SWAP_BC;

		M0 = 0x80;
		M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = MA = MB = MC = MD = ME = MF = 0;

		INPUT_BLOCK_ADD;
		A00 ^= 2;
		APPLY_P;

		SWAP_BC;
		A00 ^= 2;
		APPLY_P;

		SWAP_BC;
		A00 ^= 2;
		APPLY_P;

		SWAP_BC;
		A00 ^= 2;
		APPLY_P_FINAL;

		Hash[0] = B0;
		Hash[1] = B1;
		Hash[2] = B2;
		Hash[3] = B3;
		Hash[4] = B4;
		Hash[5] = B5;
		Hash[6] = B6;
		Hash[7] = B7;

		Hash[8] = B8;
		Hash[9] = B9;
		Hash[10] = BA;
		Hash[11] = BB;
		Hash[12] = BC;
		Hash[13] = BD;
		Hash[14] = BE;
		Hash[15] = BF;
	}
}

// #include <stdio.h>
__host__ void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce,  uint32_t *d_hash)
{
	const uint32_t threadsperblock = 64;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x14_shabal512_gpu_hash_64<<<grid, block>>>(threads, startNounce, d_hash);
}
