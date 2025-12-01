#include <cuda_runtime_api.h>
#include <cusparse.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        /*return EXIT_FAILURE;  */                                                 \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        /*return EXIT_FAILURE; */                                                \
    }                                                                          \
}

__global__ void nnzpercolSegmentReduce(int *nnzpercolinput, int *nnzpercoloutput, int nnzpercolnumber)
{
    int tid =blockDim.x * blockIdx.x +threadIdx.x;
    if(tid < nnzpercolnumber)
    {
        nnzpercoloutput[tid] = nnzpercolinput[tid+1] - nnzpercolinput[tid];
    }

}

__global__ void prefixSumKernel(int *inputarray, int *outputarray, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    outputarray[0] = 0;
    outputarray[1] = inputarray[0];


    if (idx < size && idx >1) 
    {
        for (int i = 1; i <= idx; ++i) 
        {
            outputarray[idx] += inputarray[idx-i];
        }
    }
}

void csr_to_csc(double &time_csr2csc, int m, int n, int nnz,
                const double* csr_val, const int* csr_row_ptr, const int* csr_col_ind,
                double* csc_val, int* csc_row_ind, int* csc_col_ptr, int* nnzpercol)
{
    double* d_csr_val;
    int* d_csr_row_ptr;
    int* d_csr_col_ind;
    double* d_csc_val;
    int* d_csc_row_ind;
    int* d_csc_col_ptr;
    int* d_nnzpercol;

    //int* d_nnzpercoltest;

    CHECK_CUDA (cudaMalloc((void**)&d_csr_val, nnz * sizeof(double)))
    CHECK_CUDA (cudaMalloc((void**)&d_csr_row_ptr, (m + 1) * sizeof(int)))
    CHECK_CUDA (cudaMalloc((void**)&d_csr_col_ind, nnz * sizeof(int)))
    CHECK_CUDA (cudaMalloc((void**)&d_csc_val, nnz * sizeof(double)))
    CHECK_CUDA (cudaMalloc((void**)&d_csc_row_ind, nnz * sizeof(int)))
    CHECK_CUDA (cudaMalloc((void**)&d_csc_col_ptr, (n + 1) * sizeof(int)))

    CHECK_CUDA (cudaMemcpy(d_csr_val, csr_val, nnz * sizeof(double), cudaMemcpyHostToDevice))
    CHECK_CUDA (cudaMemcpy(d_csr_row_ptr, csr_row_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA (cudaMemcpy(d_csr_col_ind, csr_col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice))

    CHECK_CUDA (cudaMalloc((void**)&d_nnzpercol, (n) * sizeof(int)))
    //CHECK_CUDA (cudaMalloc((void**)&d_nnzpercoltest, (n+1) * sizeof(int)))

    cusparseHandle_t     handle = NULL;
    cusparseStatus_t status = (cusparseCreate(&handle));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
    }

    timeval tcsr2csc1, tcsr2csc2;
    double time_csr2csc1 = 0;
    gettimeofday(&tcsr2csc1, NULL);

    size_t buffer_temp_size;
    cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, d_csr_val, d_csr_row_ptr, d_csr_col_ind,
                                  d_csc_val, d_csc_col_ptr, d_csc_row_ind, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_temp_size);
    void* buffer_temp = NULL;
    //printf("buffer_temp_size is %zd\n", buffer_temp_size);
    CHECK_CUDA(cudaMalloc(&buffer_temp, buffer_temp_size))
    CHECK_CUSPARSE(cusparseCsr2cscEx2(handle, m, n, nnz, d_csr_val, d_csr_row_ptr, d_csr_col_ind,
                                      d_csc_val, d_csc_col_ptr, d_csc_row_ind, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer_temp))
    
    gettimeofday(&tcsr2csc2, NULL);  
    time_csr2csc1 = (tcsr2csc2.tv_sec - tcsr2csc1.tv_sec) * 1000.0 + (tcsr2csc2.tv_usec - tcsr2csc1.tv_usec) / 1000.0;


    double time_csr2csc2 = 0;
    timeval tcsr2csc3, tcsr2csc4;
    gettimeofday(&tcsr2csc3, NULL);

    const int blockSize=256;
    const int numBlocks = (n+ blockSize -1 ) / blockSize;
    nnzpercolSegmentReduce<<<numBlocks, blockSize>>>(d_csc_col_ptr, d_nnzpercol, n);

    cudaDeviceSynchronize();

    gettimeofday(&tcsr2csc4, NULL); 
    time_csr2csc2 = (tcsr2csc4.tv_sec - tcsr2csc3.tv_sec) * 1000.0 + (tcsr2csc4.tv_usec - tcsr2csc3.tv_usec) / 1000.0;
    
    time_csr2csc =time_csr2csc1 + time_csr2csc2;
    //printf("------------------------------------------time_csr2csc %f \n ", time_csr2csc) ;


    //const int numBlocks1 = (n+ blockSize ) / blockSize;
    //prefixSumKernel<<<numBlocks1, blockSize>>>(d_nnzpercol, d_nnzpercoltest, n+1);

/*
    int testtt = 0;
    for(int i =0 ;i <=n; i++)
    {
        if(d_nnzpercoltest[i] != d_csc_col_ptr[i])
        {testtt++;}
    }
    if(testtt!=0)
    {printf("error------------------------------------------------\n");}
    else
    {printf("good-------------------------------------------\n");}
*/

    //printf("time_csr2csc %f \n ", time_csr2csc) ;
  

    CHECK_CUDA(cudaMemcpy(csc_val, d_csc_val, nnz * sizeof(double), cudaMemcpyDeviceToHost))
    CHECK_CUDA (cudaMemcpy(csc_row_ind, d_csc_row_ind, nnz * sizeof(int), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(csc_col_ptr, d_csc_col_ptr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(nnzpercol, d_nnzpercol, (n) * sizeof(int), cudaMemcpyDeviceToHost))

    //CHECK_CUDA(cudaMemcpy(testptr, d_nnzpercoltest, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost))


    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA(cudaFree(d_csr_val))
    CHECK_CUDA(cudaFree(d_csr_row_ptr))
    CHECK_CUDA(cudaFree(d_csr_col_ind))
    CHECK_CUDA(cudaFree(d_csc_val))
    CHECK_CUDA(cudaFree(d_csc_row_ind))
    CHECK_CUDA(cudaFree(d_csc_col_ptr))
    CHECK_CUDA(cudaFree(d_nnzpercol))
}
