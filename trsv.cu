/*----------------------------------------------------------------
    @author: Longxiang Chen, 861085404
    @email:  lchen060@ucr.edu
    @aim:    solving triangle system with leveling analysis
    @block = equation number, thread = x number
    @performace : 84 + 183 usec
----------------------------------------------------------------*/ 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
//#include <cutil.h>
#include "trsv_kernel.cu"

const int DEBUG = 0; 
extern "C"
void cpu_trsv(int size, float *L, float *b, float *x);
extern "C"
void cpu_csr_trsv(int size, float *value, int *index, int *count, float *b, float *x, int si);
// topology sorting to distribute rows into levels
void leveling(int *cind, int *ccnt, int size, int si, int *level, int *lind);


void RndM(float *A, int size, int num);
void ZeroV(float *v, int size);
void ZeroI(int *, int);
void RndV(float *, int);
// store matrix into CSR format
void csr_store(float *L, float *value, int *index, int *count, int size, int num);
// store matrix into CSC format
void csc_store(float *L, float *value, int *index, int *count, int size, int num);
int counter(float *L, int size);
void trsvOnDevice(float *b, float *x, float *val, int *index, int *count, int size, int si, int *level, int *lind);
double diff(float *x, float *xd, int size);

int main()
{
    float *L;
    float *x;
    float *cx;
    float *b;
    float *csrx;

    L = (float *)malloc(size*size*sizeof(float));
    x = (float *)malloc(size*sizeof(float));
    cx = (float *)malloc(size*sizeof(float));
    csrx = (float *)malloc(size*sizeof(float));
    b = (float *)malloc(size*sizeof(float));

    if((!L)||(!x)||(!b)||(!cx)||(!csrx))
    {
        printf("Allocate memory failed!\n");
        exit(-1);
    }

    struct timeval tstart, tend;
    int num = size*size/3;   // number of nonzero elements
    ZeroV(L, size*size);
    RndM(L, size, num);
    RndV(b, size);
    ZeroV(x, size);
    ZeroV(cx, size);

    // number of nonzero 
    int si = counter(L, size);
    printf("Matrix L[%d][%d]:\n", size, size);
    printf("The total number of nonzero elements: %d\n", si);

    
    float *val = (float *)malloc(si*sizeof(float));
    int *index = (int *)malloc(si*sizeof(int));
    int *count = (int *)malloc((size+1)*sizeof(int));
    float *cval = (float *)malloc(si*sizeof(float));
    int *cind = (int *)malloc(si*sizeof(int));
    int *ccnt = (int *)malloc((size+1)*sizeof(int));

    ZeroV(val, si);
    ZeroI(index, si);
    ZeroI(count, size);

    ZeroV(cval, si);
    ZeroI(cind, si);
    ZeroI(ccnt, size);

    csc_store(L, cval, cind, ccnt, size, si);
    ccnt[size] = si;

    csr_store(L, val, index, count, size, si);
    count[size] = si;

    int level[size];
    int lind[size];
    gettimeofday(&tstart, NULL);
    leveling(cind, ccnt, size, si, level, lind);
    gettimeofday(&tend, NULL);
    double exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on Leveling CSC format %.6lf usec\n", exe_time_cpu);

    if(DEBUG)
    {
        for(int i = 0; i < size; i++)
            printf("--> %d", level[i]);
        printf("\n");
        for(int i = 0; i < size&& lind[i] >= 0; i++)
            printf("level %d\n", lind[i]);
        printf("\n");
    }

    printf("Start on GPU...\n");
    trsvOnDevice(b, x, val, index, count, size, si, level, lind);
    printf("Start on CPU navie...\n");
    gettimeofday(&tstart, NULL);
    cpu_trsv(size, L, b, cx);
    gettimeofday(&tend, NULL);
    exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on CPU : %.6lf usec\n", exe_time_cpu);

    printf("Start on CPU CSR format...\n");
    gettimeofday(&tstart, NULL);
    cpu_csr_trsv(size, val, index, count, b, csrx, si);
    gettimeofday(&tend, NULL);
    exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on CPU CSR format: %.6lf usec\n", exe_time_cpu);

    if(DEBUG)
    {
        
        for(int i = 0; i < size*size; i++)
        {
            printf("%10.6lf\t", L[i]);
            if(i%size==size-1)
                printf("\n");
        }
        
        printf("b = \n");
        for(int i = 0; i < size; i++)
            printf("%10.8lf\t",b[i]);
            printf("\n");
        printf("cpu_x = \n");
        for(int i = 0; i < size; i++)
            printf("%10.8lf\t", cx[i]);
            printf("\n");

        printf("GPU x = \n");
        for(int i = 0; i < size; i++)
            printf("%10.8lf\t", x[i]);
        printf("\n");
    /*
        printf("csr x = \n");
        for(int i = 0; i < size; i++)
            printf("%10.8lf\t", csrx[i]);
            printf("\n");

       printf("value:\n");
       for(int i = 0; i < si; i++)
       {
           printf("%10.6lf\t", val[i]);
       }
       printf("\n");

       printf("index:\n");
       for(int i = 0; i < si; i++)
       {
           printf("%d\t", index[i]);
       }
       printf("\n");

       printf("count:\n");
       for(int i = 0; i <= size; i++)
       {
           printf("%d\t", count[i]);
       }
       printf("\n");
    */

       printf("csc_value:\n");
       for(int i = 0; i < si; i++)
       {
           printf("%10.6lf\t", cval[i]);
       }
       printf("\n");

       printf("csc_index:\n");
       for(int i = 0; i < si; i++)
       {
           printf("%d\t", cind[i]);
       }
       printf("\n");

       printf("csc_count:\n");
       for(int i = 0; i <= size; i++)
       {
           printf("%d\t", ccnt[i]);
       }
       printf("\n");

        double err = diff(cx, csrx, size);
        //err += diff(x, csrx, size);
        if (err < 10E-3)
            printf("Correct results between CPU and CPU_CSR! Norm error =  %e\n", err);
            //printf("Speedup: %.3lf\n", exe_time_cpu/exe_time);
        else
            printf("Error results between CPU and CPU_CSR! Error : %e\n", err);
        double err1 = diff(x, csrx, size);
        if (err < 10E-3)
            printf("Correct results between CPU and GPU! Norm error =  %e\n", err1);
            //printf("Speedup: %.3lf\n", exe_time_cpu/exe_time);
        else
            printf("Error results between CPU and GPU! Error : %e\n", err1);
    }


    free(L);
    free(x);
    free(cx);
    free(csrx);
    free(b);
    free(val);
    free(index);
    free(count);
    free(cval);
    free(ccnt);
    free(cind);
}

void trsvOnDevice(float *b, float *x, float *val, int *index, int *count, int size, int si, int *level, int *lind)
{
    float *vald = val;
    int *indexd = index;
    int *countd = count;
    int *leveld = level;
    int *lindd = lind;
    float *bd = b;
    float *xd = x;

    int nl = 0;
    for(int i = 0; i < size && lind[i] >=0; i++)
    {
        nl++;
    }
    nl--;

    cudaMalloc((void**)&vald, si*sizeof(float));
    cudaMalloc((void**)&indexd, si*sizeof(int));
    cudaMalloc((void**)&countd, size*sizeof(int));
    cudaMalloc((void**)&bd, size*sizeof(float));
    cudaMalloc((void**)&xd, size*sizeof(float));
    cudaMalloc((void**)&lindd, size*sizeof(int));
    cudaMalloc((void**)&leveld, size*sizeof(int));

    cudaMemcpy(vald, val, si*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(indexd, index, si*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(countd, count, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xd, x, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lindd, lind, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(leveld, level, size*sizeof(int), cudaMemcpyHostToDevice);

    struct timeval tstart, tend;	
    gettimeofday(&tstart, NULL);

    dim3 dimBlock(size, 1);
    for(int iter = 0; iter < nl; iter++)
    {
	int bk = lind[iter+1] - lind[iter];
        dim3 dimGrid(bk, 1);
	/*
	printf("this level has %d topo.\n", bk);
	for(int i = level[lind[iter]]; i < level[lind[iter]] + bk; i++)
		printf("row: %d\n", i);
	printf("\n");
	*/
    	trsvKernel<<<dimGrid, dimBlock>>>(bd, xd, vald, indexd, countd, leveld, lindd, iter);
    }

    gettimeofday(&tend, NULL);
    double exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on GPU %.6lf usec\n", exe_time_cpu);




    cudaMemcpy(x, xd, size*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(vald);
    cudaFree(indexd);
    cudaFree(countd);
    cudaFree(bd);
    cudaFree(xd);
    cudaFree(leveld);
    cudaFree(lindd);
}


double diff(float *x, float *xd, int size)
{
    double sum = 0.0, norm = 0.0;;
    for(int i = 0; i < size; i++)
    {
    //printf("x = %lf, xd = %lf\n", x[i], xd[i]);
        sum += (x[i]-xd[i])*(x[i]-xd[i]);
        norm += xd[i]*xd[i];
    }
    return (sqrt(sum)/sqrt(norm));
}


int counter(float *L, int size)
{
    int sum = 0;
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            if(fabs(L[i*size + j]) > 10E-6)
                sum++;
        }
    }
    return sum;
}


void csr_store(float *L, float *value, int *index, int *count, int size, int num)
{
    int k = 0, p = 0, row = -1;
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            if(fabs(L[i*size + j]) > 10E-6)
            {
                index[k] = j;
                value[k] = L[i*size + j];
                if(i > row)  // new line
                {
                    count[p] = k;
                    row = i;
                    p++;
                }
                k++;
            }
        }
    }
}

void csc_store(float *L, float *value, int *index, int *count, int size, int num)
{
    int k = 0, p = 0, col = -1;
    for(int i = 0; i < size; i++)
    {
        for(int j = i; j < size; j++)
        {
            if(fabs(L[j*size + i]) > 10E-6)
            {
                index[k] = j;
                value[k] = L[j*size + i];
                if(i > col)  // new line
                {
                    count[p] = k;
                    col = i;
                    p++;
                }
                k++;
            }
        }
    }
}



void RndM(float *A, int size, int num)
{
    for(int i = 0; i < size; i++)
            A[i*size + i] = 1.0;//rand() / (float)RAND_MAX;
    for(int i = 1; i < size; i++) 
    {
        for(int j = 0; j < 5 &&j < i; j++)
        {
            int idx = (rand())%(i) - j ;
            A[i*size + idx] = rand() / (float)RAND_MAX / 1000.0;
        }
    }
}

void ZeroV(float *v, int size)
{
    for(int i = 0; i < size; i++)
        v[i] = 0.0;
}

void ZeroI(int *v, int size)
{
    for(int i = 0; i < size; i++)
        v[i] = 0;
}

void RndV(float *v, int size)
{
    for(int i = 0; i < size; i++)
        v[i] = rand()*3 / (float)RAND_MAX;
}

void leveling(int *cind, int *ccnt, int size, int si, int *level, int *lind)
{
    int count = 0;
    int indeg[size];
    for(int i = 0; i < size; i++)
    {
        indeg[i] = -1;
    }
        
    for(int i = 0; i < si; i++)
    {
        indeg[cind[i]]++;   // get indegree of each row
    }

    int ind = 0;
    int num_lev = 0;
    lind[0] = 0;
    num_lev++;
    while(count < size)
    {
        ind = count;
        for(int i = 0; i < size; i++)
        {
            if(indeg[i] == 0)
            {
                level[count] = i;
//		printf("--> %d\n", i);
                count++;
                indeg[i] = -1;
            }
        }
        lind[num_lev++] = count;
        for(int j = ind; j < count; j++)
        {
            for(int k = 1; k < ccnt[level[j]+1] - ccnt[level[j]]; k++)
            {
                indeg[cind[ccnt[level[j]]+k]]--;
            }
        }
    }
    lind[num_lev++] = -1; 

    return ;
}
