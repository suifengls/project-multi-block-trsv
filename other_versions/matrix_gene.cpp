#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
using namespace std;

void ZeroV(float *v, int size)
{
    for(int i = 0; i < size; i++)
        v[i] = 0.0;
}

void RndV(float *v, int size)
{
    for(int i = 0; i < size; i++)
        v[i] = rand()*3 / (float)RAND_MAX;
}

void RndM(float *A, int size, int num)
{
    for(int i = 0; i < size; i++)
            A[i*size + i] = 1.0;//rand() / (float)RAND_MAX;
    for(int i = 1; i < size; i++) 
    {
        for(int j = 0; j < 5 && j < i ; j++)
        {
            int idx = (rand())%(i) - j ;
            A[i*size + idx] = rand() / (float)RAND_MAX / 1000.0;
        }
    }
}

int main()
{
    int size = 1000;
    int num = size*size*10; // number of nonzero elements
    float *L;
    float *b;
    L = (float *)malloc(size*size*sizeof(float));
    b = (float *)malloc(size*sizeof(float));
    ZeroV(L, size*size);
    ZeroV(b, size);
    RndM(L, size, num);
    RndV(b, size);
   
    
    ofstream outfile("L.txt");
    outfile << size << "\t" << size << endl;
    for(int i = 0; i < size*size; i++)
            outfile << L[i] << "\t";
    outfile.close();
    

    ofstream outfile0("b.txt");
    outfile0 << size << "\t" << "1" <<  endl;
    for(int i = 0; i < size; i++)
            outfile0 << b[i] << "\t";
    outfile0.close();

    free(L);
    free(b);
    return 0;
}


