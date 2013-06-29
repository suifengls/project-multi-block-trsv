#ifndef _FTMM_KERNEL_H_
#define _FTMM_KERNEL_H_

#include <stdio.h>
#include "trsv.h"

__global__ void trsvKernel(float *b, float *x, float *val, int *ind, int *cnt, int *topo, int *lv, int nl)
{
    const int tid = threadIdx.x;
    for(int i = 0; i < nl; i++)
    {
        for(int j = lv[i]; j < lv[i+1]; j++)
        {
            if(tid == topo[j])
            {
                x[tid] = b[tid] / val[cnt[tid]];
                int cc = cnt[tid+1] - cnt[tid];
                for(int k = cnt[tid]+1; k < cnt[tid]+cc; k++)
                {
                    b[ind[k]] -= x[tid] * val[k];
                }
            }
        }
        __syncthreads();
    }
}
#endif
