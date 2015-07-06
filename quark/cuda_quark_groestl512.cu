// Auf QuarkCoin spezialisierte Version von Groestl inkl. Bitslice

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#define TPB 512
#define THF 4

// 64 Register Variante für Compute 3.0
#include "groestl_functions_quad.cu"
#include "bitslice_transformations_quad.cu"

__global__ __launch_bounds__(TPB, 2)
void quark_groestl512_gpu_hash_64_quad(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	uint32_t msgBitsliced[8];
	uint32_t state[8];
	uint32_t hash[16];
	// durch 4 dividieren, weil jeweils 4 Threads zusammen ein Hash berechnen
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
    if (thread < threads)
    {
        // GROESTL
        uint32_t nounce = g_nonceVector ? g_nonceVector[thread] : (startNounce + thread);
		uint32_t hashPosition = nounce - startNounce;
        uint32_t *inpHash = &g_hash[hashPosition * 16];

        const uint32_t thr = threadIdx.x & (THF-1);

		uint32_t message[8] =
		{
			inpHash[thr], inpHash[(THF)+thr], inpHash[(2 * THF) + thr], inpHash[(3 * THF) + thr],0, 0, 0, 
		};
		if (thr == 0) message[4] = 0x80UL;
		if (thr == 3) message[7] = 0x01000000UL;

		to_bitslice_quad(message, msgBitsliced);

        groestl512_progressMessage_quad(state, msgBitsliced);

		from_bitslice_quad(state, hash);

		if (thr != 0) return;

		#pragma unroll
		for (int k = 0; k < 16; k++) inpHash[k] = hash[k];
    }
}

__host__ void quark_groestl512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
    // Compute 3.0 benutzt die registeroptimierte Quad Variante mit Warp Shuffle
    // mit den Quad Funktionen brauchen wir jetzt 4 threads pro Hash, daher Faktor 4 bei der Blockzahl
    const int factor = THF;

    // berechne wie viele Thread Blocks wir brauchen
	dim3 grid(factor*((threads + TPB - 1) / TPB));
	dim3 block(TPB);

    quark_groestl512_gpu_hash_64_quad<<<grid, block>>>(threads, startNounce, d_hash, d_nonceVector);
}

