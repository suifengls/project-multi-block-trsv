#ifndef _FTMM_KERNEL_H_
#define _FTMM_KERNEL_H_

#include <stdio.h>
#include <assert.h>
#include "trsv.h"

// CSR Format
__global__ void trsvKernel(float *b, float *x, float *val, int *ind, int *cnt, int *topo, int *lv, int iter)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
    	float __shared__ dot[size];
	int row = topo[lv[iter]+bid];
	int nz = cnt[row+1] - cnt[row];
	
	if(nz == 1) // only one element
	{
		x[row] = b[row]/val[cnt[row]];
		return;
	}
	else
	{
    		float tmpval, tmpx;
		if(tid < nz-1)
		{
			tmpval = val[cnt[row]+tid];
			tmpx = x[ind[cnt[row]+tid]];
			dot[tid] = tmpval*tmpx;
		}
		else
			dot[tid] = 0.0;
	}
	__syncthreads();

	// reduce
	int tid2;
	int tot = ceilf(log2f((nz-1)*1.0));
	while(tot > 1)
	{
		int offset = tot*2;
		if(tid < offset)
		{
			tid2 = tid + offset;
			if(tid2 < cnt[bid+1]-cnt[bid]-1)
				dot[tid] += dot[tid2];
		}
		__syncthreads();
		tot = offset;
	}

	if(tid == nz-1)
	{
		x[row] = (b[row] - dot[0]) / val[cnt[row+1]-1];
	}
}
#endif
