#ifndef _FTMM_KERNEL_H_
#define _FTMM_KERNEL_H_

#include <stdio.h>
#include "trsv.h"

// CSR Format
__global__ void trsvKernel(float *b, float *x, float *val, int *ind, int *cnt, int *topo, int *lv, int nl)
{
    const int tid = threadIdx.x;
    for(int i = 0; i < nl; i++)
    {
        for(int j = lv[i]; j < lv[i+1]; j++)
        {
            if(tid == topo[j])
            {
		for(int k = cnt[tid]; k < cnt[tid+1]-1; k--)
                {
                	b[tid] -= x[ind[k]] * val[k];
                }
                x[tid] = b[tid];// / val[cnt[tid+1]-1];
            }
        }
        __syncthreads();
    }
}
#endif
