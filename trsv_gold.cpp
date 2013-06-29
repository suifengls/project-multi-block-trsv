#include <stdio.h>
#include <math.h>

extern "C"
void cpu_trsv(int size, float *L, float *b, float *x);
extern "C"
void cpu_csr_trsv(int size, float *value, int *index, int *count, float *b, float *x, int si);

void cpu_trsv(int size, float *L, float *b, float *x)
{
    x[0] = b[0]/L[0];
    for(int i = 1; i < size; i++)
    {
        float sum = 0.0;
        for(int j = 0; j < i; j++)
        {
            sum += x[j] * L[i*size + j];
        }
        x[i] = (b[i] - sum) / L[i*size + i];
	//printf("b -sum = %lf, Lii = %lf, cpu x[%d] = %lf\n", b[i]-sum, L[i*size+i], i, x[i]);
    }
}


// si = number of nonzero elements
void cpu_csr_trsv(int size, float *val, int *ind, int *cnt, float *b, float *x, int si)
{
    x[0] = b[0] / val[0];
    for(int i = 1; i < size; i++)
    {
        int num = cnt[i+1] - cnt[i];
        if(num == 1)
        {
            x[i] = b[i] / val[cnt[i]];
            continue;
        }
        else
        {
            float sum = 0.0;
            for(int j = 0; j < num - 1; j++)
            {
                sum += val[cnt[i] + j] * x[ind[cnt[i] + j]];
            }
            //x[ind[cnt[i] + num - 1]] = (b[i] - sum) / val[cnt[i] + num - 1];
            x[i] = (b[i] - sum) / val[cnt[i] + num - 1];
            //printf("i = %d, x = %.3lf\n", i, x[i]);
        }
    }
}
