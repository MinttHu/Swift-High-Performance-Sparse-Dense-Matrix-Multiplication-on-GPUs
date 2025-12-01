#include <iostream>
#include <vector>
#include "sys/time.h"
#include "auxiliary.h"


#define BN_DEFAULT 128

#ifndef VERIFYCSC
#define VERIFYCSC 1
#endif

void generate_uniform_csr(int M, int K, int nnz_total,
                          int* row_ptr, int* col_idx)
{
    int avg_nnz = nnz_total / M;
    int remaining = nnz_total;

    srand(42);

    row_ptr[0] = 0;
    int idx = 0;

    for (int i = 0; i < M; ++i)
    {
        int nnz_row = std::min(avg_nnz, remaining - (M - i - 1));
        nnz_row = std::max(nnz_row, 1);
        remaining -= nnz_row;

    
        int cols[100]; 
        for (int j = 0; j < K; ++j) cols[j] = j;

        for (int j = 0; j < nnz_row; ++j)
        {
            int r = j + rand() % (K - j);
            int tmp = cols[j];
            cols[j] = cols[r];
            cols[r] = tmp;
        }

        row_ptr[i+1] = idx;
    }
}

void generate_concentrated_csr(int M, int K, int nnz_total,
                          int* row_ptr, int* col_idx)
{
    srand(43);

    int* row_weights = new int[M];
    int* nnz_per_row = new int[M];
    int sum_weight = 0;

    for (int i = 0; i < M; ++i)
    {
        row_weights[i] = 1 + rand() % 10; 
        sum_weight += row_weights[i];
    }

    int assigned_total = 0;
    for (int i = 0; i < M; ++i)
    {
        nnz_per_row[i] = std::max(1, (row_weights[i] * nnz_total) / sum_weight);
        assigned_total += nnz_per_row[i];
    }

    int diff = nnz_total - assigned_total;
    for (int i = 0; i < abs(diff); ++i)
    {
        int idx = i % M;
        nnz_per_row[idx] += (diff > 0 ? 1 : -1);
        if (nnz_per_row[idx] < 1) nnz_per_row[idx] = 1;
    }

    row_ptr[0] = 0;
    int idx = 0;
    int* cols = new int[K];
    for (int i = 0; i < K; ++i) cols[i] = i;

    for (int i = 0; i < M; ++i)
    {
        int nnz_row = nnz_per_row[i];

        for (int j = 0; j < nnz_row; ++j)
        {
            int r = j + rand() % (K - j);
            int tmp = cols[j]; cols[j] = cols[r]; cols[r] = tmp;
        }

        for (int j = 0; j < nnz_row; ++j)
            col_idx[idx++] = cols[j];

        row_ptr[i+1] = idx;
    }

    delete[] row_weights;
    delete[] nnz_per_row;
    delete[] cols;
}



__global__ void csr_spmm_kernel(int* row_ptr,
                                int*  col_idx,
                                double* values,
                                double* B,
                                double* C,
                                int M, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= M) return;

    int row_start = row_ptr[row];
    int row_end   = row_ptr[row+1];

    for(int col = 0; col < N; ++col){
        double sum = 0.0f;
        for(int idx = row_start; idx < row_end; ++idx){
            int k = col_idx[idx];
            sum += values[idx] * B[k * N + col]; // B in row-major
        }
        C[row * N + col] = sum;
    }
}


__global__ void csr_spmm_kernel2(int* row_ptr,
                                int*  col_idx,
                                double* values,
                                double* B,
                                double* C,
                                int M, int K, int N)
{
    int rid = blockDim.y * blockIdx.x + threadIdx.y;
    int dn_index = blockIdx.y << 5;
    if(rid < M)
    {
        int len = row_ptr[rid+1] - row_ptr[rid];
        for(int kk = 0; kk < 32 ; kk++)
        {
          double resultval = 0.0f;
          for(int i = threadIdx.x; i< len; i+= 32)
          {
            int cidx = col_idx[row_ptr[rid] + i];
            double val = values[row_ptr[rid] + i];
            double denseval = B[cidx * N + dn_index + kk];
            resultval += val * denseval;
          }
          for (int offset = 16; offset > 0; offset >>= 1)
          resultval += __shfl_down_sync(0xffffffff, resultval, offset);

         if ((threadIdx.x & 31) == 0)
          C[rid * N + dn_index + kk] = resultval;  
        }
    }
}


int main(int argc, char ** argv)
{
    if(argc <2)
    {
        printf("error order\n");
        return 0;
    }

    int device_id = 0;
    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }

    int BN = BN_DEFAULT;

    if (argc > 4) {
        BN = std::stoi(argv[4]); 
    }

    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);


    printf("---------------------------------------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);


	char  *filename;
    filename = argv[3];


    int rowA,colA,nnz;
    int isSymmetricA;
    double *csrval;
    int *csrrowptr;
    int *csrcolidx;

    mmio_allinone(&rowA, &colA, &nnz, &isSymmetricA, &csrrowptr, &csrcolidx, &csrval ,filename);

//|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-
    std::string filePath(filename);

    size_t pos = filePath.find_last_of("/\\");
    std::string fileName;
    if (pos != std::string::npos) {
        fileName = filePath.substr(pos + 1);
    } else {
        fileName = filePath;
    }

    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos) {
        fileName = fileName.substr(0, dotPos);
    }


    std::snprintf(filename, 256, "%s", fileName.c_str());

//|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-

    printf("read success,input matrix: %s :(%i,%i) nnz =%i  width=%d \n",filename,rowA,colA,nnz,BN);
  

    double *dense_matrix = (double *)malloc(sizeof(double) * colA*BN);
    memset(dense_matrix,0, sizeof(double) * colA*BN);

    unsigned seed;
    seed = time(0);
    srand(seed);
    for (int i=0; i<colA*BN; i++) 
    {
        dense_matrix[i] = double(rand() %100 - 50)/100;
    }

    for (int i=0; i<nnz; i++) 
    {
        csrval[i] = double(rand() %1000 - 500)/1000;
    }


    int *row_ptr_uniform = (int *)malloc(sizeof(int) * (rowA+1));
    memset(row_ptr_uniform,0, sizeof(int) * (rowA+1));
    int *col_idx_uniform = (int *)malloc(sizeof(int) * nnz);
    memset(col_idx_uniform,0, sizeof(int) * nnz);   



    int *row_ptr_conc = (int *)malloc(sizeof(int) * (rowA+1));  
    memset(row_ptr_conc,0, sizeof(int) * (rowA+1));
    int *col_idx_conc = (int *)malloc(sizeof(int) * nnz);
    memset(col_idx_conc,0, sizeof(int) * nnz);

    generate_uniform_csr(rowA, colA, nnz, row_ptr_uniform, col_idx_uniform);

    generate_concentrated_csr(rowA, colA, nnz, row_ptr_conc, col_idx_conc);


    double *golden_matrix_c1 = (double *)malloc(sizeof(double)*rowA *BN);
    memset(golden_matrix_c1,0, sizeof(double) * rowA*BN);  


    for(int i=0; i<rowA;i++)
    {
      for(int j=row_ptr_uniform[i]; j<row_ptr_uniform[i+1]; j++)
      {
        for(int k=0; k<BN; k++)
        {
          int dense_index = col_idx_uniform[j] *  BN;
          golden_matrix_c1[i*BN + k] += csrval[j] * dense_matrix[dense_index + k];
        }
      }
    }


    double *golden_matrix_c2 = (double *)malloc(sizeof(double)*rowA *BN);
    memset(golden_matrix_c2,0, sizeof(double) * rowA*BN);   
    for(int i=0; i<rowA;i++)
    {
      for(int j=row_ptr_conc[i]; j<row_ptr_conc[i+1]; j++)
      {
        for(int k=0; k<BN; k++)
        {
          int dense_index = col_idx_conc[j] *  BN;
          golden_matrix_c2[i*BN + k] += csrval[j] * dense_matrix[dense_index + k];
        }
      }
    }

    double *golden_matrix_c3 = (double *)malloc(sizeof(double)*rowA *BN);
    memset(golden_matrix_c3,0, sizeof(double) * rowA*BN);   
    for(int i=0; i<rowA;i++)
    {
      for(int j=csrrowptr[i]; j<csrrowptr[i+1]; j++)
      {
        for(int k=0; k<BN; k++)
        {
          int dense_index = csrcolidx[j] *  BN;
          golden_matrix_c3[i*BN + k] += csrval[j] * dense_matrix[dense_index + k];
        }
      }
    }

    double *d_resultmtx_uniform;
    double *d_resultmtx2_conc;
    double *d_resultmtx_original;
    
    double *d_valuesA;
    int *d_row_ptr_uniform;
    int *d_col_idx_uniform;
    
    int *d_row_ptr_conc;
    int *d_col_idx_conc;

    int *d_row_ptr_original;
    int *d_col_idx_original;

    double *d_densemtx;

    double *h_resultmtx_uniform = (double *)malloc(sizeof(double) * rowA*BN);
    memset(h_resultmtx_uniform,0, sizeof(double) * rowA*BN);
    double *h_resultmtx2_conc = (double *)malloc(sizeof(double) * rowA*BN);
    memset(h_resultmtx2_conc,0, sizeof(double) * rowA*BN);

    double *h_resultmtx_original = (double *)malloc(sizeof(double) * rowA*BN);
    memset(h_resultmtx_original,0, sizeof(double) * rowA*BN);

    cudaMalloc((void **)&d_resultmtx_uniform, (rowA * BN) * sizeof(double));
    cudaMalloc((void **)&d_resultmtx2_conc, (rowA * BN) * sizeof(double));
    cudaMalloc((void **)&d_resultmtx_original, (rowA * BN) * sizeof(double));

    cudaMalloc((void **)&d_valuesA, nnz * sizeof(double));

    cudaMalloc((void **)&d_row_ptr_uniform, (rowA + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx_uniform, nnz * sizeof(int));

    cudaMalloc((void **)&d_row_ptr_conc, (rowA + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx_conc, nnz * sizeof(int));

    cudaMalloc((void **)&d_row_ptr_original, (rowA + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx_original, nnz * sizeof(int));

    cudaMalloc((void **)&d_densemtx, (colA * BN) * sizeof(double));

    cudaMemcpy(d_valuesA, csrval, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr_uniform, row_ptr_uniform, (rowA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx_uniform, col_idx_uniform, nnz * sizeof(int), cudaMemcpyHostToDevice);     
    cudaMemcpy(d_row_ptr_conc, row_ptr_conc, (rowA + 1) * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_col_idx_conc, col_idx_conc, nnz * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_row_ptr_original, csrrowptr, (rowA + 1) * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_col_idx_original, csrcolidx, nnz * sizeof(int), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_densemtx, dense_matrix, (colA * BN) * sizeof(double), cudaMemcpyHostToDevice); 
    
    int num_threads =  32*2;
    int num_blocks = ceil(( double)rowA / (double)2); 

    int block = 128;
    int grid = (rowA + block - 1)/block;

    float time_orignal = 0.0f;
    cudaEvent_t event5,event6;
    cudaEventCreate(&event5);       
    cudaEventCreate(&event6);
    cudaDeviceSynchronize();
    cudaEventRecord(event5,0);
    csr_spmm_kernel2<<<dim3((rowA+8-1)/8,(BN+31)/32,1),dim3(32,8,1),32*8*(sizeof(int)+sizeof(double)),0>>>(d_row_ptr_original, d_col_idx_original, d_valuesA, d_densemtx, d_resultmtx_original, rowA, colA, BN);
    cudaDeviceSynchronize();
    cudaEventRecord(event6,0);
    cudaEventSynchronize(event5);
    cudaEventSynchronize(event6);   
    cudaEventElapsedTime(&time_orignal, event5, event6);
    cudaDeviceSynchronize();

    float time_conc = 0.0f;
    cudaEvent_t event3,event4;
    cudaEventCreate(&event3);
    cudaEventCreate(&event4);
    cudaDeviceSynchronize();
    cudaEventRecord(event3,0);
    csr_spmm_kernel2<<<dim3((rowA+8-1)/8,(BN+31)/32,1),dim3(32,8,1),32*8*(sizeof(int)+sizeof(double)),0>>>(d_row_ptr_conc, d_col_idx_conc, d_valuesA, d_densemtx, d_resultmtx2_conc, rowA, colA, BN);
    //csr_spmm_kernel<<<grid, block>>>(d_row_ptr_conc, d_col_idx_conc, d_valuesA, d_densemtx, d_resultmtx2_conc, rowA, colA, BN);
    cudaDeviceSynchronize();

    cudaEventRecord(event4,0);
    cudaEventSynchronize(event3);
    cudaEventSynchronize(event4);
    cudaEventElapsedTime(&time_conc, event3, event4);
    cudaDeviceSynchronize();



    float time_uniform = 0.0f;

    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    csr_spmm_kernel2<<<dim3((rowA+8-1)/8,(BN+31)/32,1),dim3(32,8,1),32*8*(sizeof(int)+sizeof(double)),0>>>(d_row_ptr_uniform, d_col_idx_uniform, d_valuesA, d_densemtx, d_resultmtx_uniform, rowA, colA, BN);
    cudaDeviceSynchronize();

    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&time_uniform, event1, event2);
    cudaDeviceSynchronize();


    cudaMemcpy(h_resultmtx_uniform, d_resultmtx_uniform, (rowA * BN) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resultmtx2_conc, d_resultmtx2_conc, (rowA * BN) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_resultmtx_original, d_resultmtx_original, (rowA * BN) * sizeof(double), cudaMemcpyDeviceToHost);

    int errorunify=0;
    ResultVerify(h_resultmtx_uniform, golden_matrix_c1, rowA*BN, errorunify);
    if(errorunify !=0)
    {
        printf("error spmm unify, error = %d\n", errorunify); 
    }
    else
    {
        printf("success spmm (load balance) time: %f ms\n", time_uniform);
    }

    int errorconc=0;
    ResultVerify(h_resultmtx2_conc, golden_matrix_c2, rowA*BN, errorconc);
    if(errorconc !=0)       
    {
        printf("error spmm concentrated, error = %d\n", errorconc); 
    }
    else
    {
        printf("success spmm (load imbalance) time: %f ms\n", time_conc);
    }  
    
    int errororiginal=0;
    ResultVerify(h_resultmtx_original, golden_matrix_c3, rowA*BN, errororiginal);
    if(errororiginal !=0)    
    {       
        printf("error spmm original, error = %d\n", errororiginal); 
    }
    else
    {
        printf("success spmm original time: %f ms\n", time_orignal);
    }   

    if(errorunify ==0 && errorconc==0 && errororiginal==0)
    {

        FILE *fout = fopen(RESULT_FILE_PATH, "a");
        if (fout == NULL)
        printf("Writing results fails.\n");
            fprintf(fout, "%s Width %d Load_balance %f Load_imbalance %f \n",
               filename, BN , time_uniform, time_conc);
        fclose(fout);  
    }


}