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
void topological(int a[][size]);
void leveling(int *cind, int *ccnt, int size, int si, int *level, int *lind);
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
    for(int i = 0; i < size; i++)
    {
        //printf("deg = %d\n", indeg[i]);
    }

    int ind = 0;
    while(count < size)
    {
                ind = count;
        for(int i = 0; i < size; i++)
        {
            if(indeg[i] == 0)
            {
                level[count] = i;
                count++;
                indeg[i] = -1;
            }
        }
        for(int j = ind; j < count; j++)
        {
            for(int k = 1; k < ccnt[level[j]+1] - ccnt[level[j]]; k++)
            {
                indeg[cind[ccnt[level[j]]+k]]--;
            }
        }
    }
    //for(int i = 0; i < size; i++)
    //    printf("--> %d\n", level[i]);

    return ;
}


void RndM(float *A, int size, int num);
void ZeroV(float *v, int size);
void ZeroI(int *, int);
void RndV(float *, int);
void csr_store(float *L, float *value, int *index, int *count, int size, int num);
void csc_store(float *L, float *value, int *index, int *count, int size, int num);
int counter(float *L, int size);
void trsvOnDevice(float *b, float *x, float *val, int *index, int *count, int size, int si);
int NearPowTwo(int n);
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
    int num = size*size;   // number of nonzero elements
    ZeroV(L, size*size);
    RndM(L, size, num);
    RndV(b, size);
    ZeroV(x, size);
    ZeroV(cx, size);

    // number of nonzero 
    int si = counter(L, size);
    printf("Matrix L[%d][%d]:\n", size, size);
    printf("The total number of nonzero elements: %d\n", si);

    float val[si];  // value of nonzero elements
    int index[si];  // column index of each element
    int count[size + 1];        // first nonzero element in value arra
    ZeroV(val, si);
    ZeroI(index, si);
    ZeroI(count, size);

    float cval[si];  // value of nonzero elements
    int cind[si];  // column index of each element
    int ccnt[size + 1];        // first nonzero element in value arra
    ZeroV(cval, si);
    ZeroI(cind, si);
    ZeroI(ccnt, size);

    csc_store(L, cval, cind, ccnt, size, si);
    ccnt[size] = si;

    csr_store(L, val, index, count, size, si);
    count[size] = si;

    int level[si];
    int lind[si];
    gettimeofday(&tstart, NULL);
    leveling(cind, ccnt, size, si, level, lind);
    gettimeofday(&tend, NULL);
    double exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on Leveling CSC format %.6lf msec\n", exe_time_cpu/1000.0);
    // Analyze DAG
    int a[size][size];
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
            a[i][j] = 0;
    }
    for(int i = 0; i < size; i++)
    {
        if(count[i+1] - count[i] == 1)
        {
            continue;
        }
        else // have dependence
        {
            for(int j = 0; j < count[i+1] - count[i] - 1; j++)
            {
                a[index[count[i] + j]][i] = 1;
                if(DEBUG)
                    printf("%d --> %d\n", index[count[i]+j], i);
            }
        }
    }
    if(DEBUG)
    {
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
                printf("%d\t",a[i][j]);
            printf("\n");    
        }
    }

    //topology sorting and leveling
    gettimeofday(&tstart, NULL);
    //topological(a);
    gettimeofday(&tend, NULL);
    exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on Topology Sorting AM format %.6lf msec\n", exe_time_cpu/1000.0);

    printf("Start on GPU...\n");
    //trsvOnDevice(b, x, val, index, count, size, si);
    printf("Start on CPU navie...\n");
    gettimeofday(&tstart, NULL);
    cpu_trsv(size, L, b, cx);
    gettimeofday(&tend, NULL);
    exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on CPU CSR format: %.6lf msec\n", exe_time_cpu/1000.0);

    printf("Start on CPU CSR format...\n");
    gettimeofday(&tstart, NULL);
    cpu_csr_trsv(size, val, index, count, b, csrx, si);
    gettimeofday(&tend, NULL);
    exe_time_cpu = (tend.tv_sec * 1000000 + tend.tv_usec) - (tstart.tv_sec * 1000000 + tstart.tv_usec);
    printf("Execution time on CPU CSR format: %.6lf msec\n", exe_time_cpu/1000.0);

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
    }

    double err = diff(cx, csrx, size);
    //err += diff(x, csrx, size);
    if (err < 10E-6)
        printf("Correct results! Norm error =  %e\n", err);
        //printf("Speedup: %.3lf\n", exe_time_cpu/exe_time);
    else
        printf("Error results from CPU! Error : %e\n", err);

    free(L);
    free(x);
    free(cx);
    free(b);
}

void trsvOnDevice(float *b, float *x, float *val, int *index, int *count, int size, int si)
{
    float *vald = val;
    int *indexd = index;
    int *countd = count;
    float *bd = b;
    float *xd = x;
    cudaMalloc((void**)&vald, si*sizeof(float));
    cudaMalloc((void**)&indexd, si*sizeof(int));
    cudaMalloc((void**)&countd, size*sizeof(int));
    cudaMalloc((void**)&bd, size*sizeof(float));
    cudaMalloc((void**)&xd, size*sizeof(float));

    cudaMemcpy(vald, val, si*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(indexd, index, si*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(countd, count, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xd, x, size*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // start from row by row
    for(int iter = 0; iter < size; iter++)
    {
        int block;
        dim3 dimGrid(1, 1);
        block = count[iter+1] - count[iter];
        int near = NearPowTwo(block);
        dim3 dimBlock(block, 1);
        //trsvKernel<<<dimGrid, dimBlock>>>(bd, xd, vald, indexd, countd, iter, near);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float exetime;
    cudaEventElapsedTime(&exetime, start, stop);
    printf("Execution time on GPU: %.6lf msec\n", exetime);


    cudaMemcpy(x, xd, size*sizeof(float), cudaMemcpyDeviceToHost);
    if(DEBUG)
    {
        printf("GPU x = \n");
        for(int i = 0; i < size; i++)
            printf("%10.8lf\t", x[i]);
        printf("\n");
    }

    cudaFree(vald);
    cudaFree(indexd);
    cudaFree(countd);
    cudaFree(bd);
    cudaFree(xd);
}


double diff(float *x, float *xd, int size)
{
    double sum = 0.0, norm = 0.0;;
    for(int i = 0; i < size; i++)
    {
        sum += (x[i]-xd[i])*(x[i]-xd[i]);
        norm += x[i]*x[i];
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
        for(int j = i; j <= size; j++)
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
            A[i*size + i] = rand()*3 / (float)RAND_MAX;
    int k = num;        
    for(int i = 1; i < size && k > 0; i++) 
    {
        int idx = rand()%size;
        int idy = rand()%size;
        if(idx > idy)
            A[idx*size + idy] = rand()*3 / (float)RAND_MAX;
        else
            A[idy*size + idx] = rand()*3 / (float)RAND_MAX;
        k--;    
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

int NearPowTwo(int n)
{
    if(!n) return n;

    int x = 1;
    while(x < n)
    {
        x <<= 1;
    }
    return x;
}

void topological(int a[][size])
{
    int level[size], indeg[size], flag[size], count = 0;
    for(int i = 0; i < size; i++)
    {
        level[i] = 0;
        indeg[i] = 0;
        flag[i] = 0;
    }
    int max = 0;
    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
        {
            indeg[i] = indeg[i] + a[j][i];
            level[i] = indeg[i];
            if(indeg[i] > max)
                max = indeg[i];
        }
    //printf("Max level : %d\n", max);

    //printf("Topo sorting:\n");
    int lv = 1;
    while(count < size)
    {
        for(int k = 0; k < size; k++)
        {
            //printf("Level %d:\n", k);
            if((indeg[k] == 0) && (flag[k] == 0))
            {
    //            printf("%d --> ", k);
                flag[k] = lv++;
            }
            for(int i = 0; i < size; i++)
            {
                if(a[i][k] == 1) indeg[k]--;
            }
        }
        count++;
    }
    //printf("\n");
    for(int i = 1; i <= size; i++)
    {
        for(int k = 0; k < size; k++)
        {
            if(flag[k] == i)
            {
                int max = -1;
                for( int j = 0; j <= k; j++)
                {
                    if(a[j][k] != 0 && level[j] > max)
                        max = level[j];
                }
                level[k] = max + 1;
            }
        }
    }

    //for(int i = 0; i < size; i++)
    //    printf("row %d, topo: %d, level %d\n", i, flag[i], level[i]);
    return ;
}
