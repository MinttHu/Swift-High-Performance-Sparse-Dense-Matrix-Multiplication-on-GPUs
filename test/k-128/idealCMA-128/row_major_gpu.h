#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA slide Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA slide Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}


__global__ void Swift_SpMM_kernel(int rowA, int colA, int width, int nnz,
                                     double *d_matrixA,
                                     double *d_densemtx,
                                     double *d_resultmtx)
{
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    const int th_tile = global_id >> 5;
    const int lane_id = (TILESIZE -1) & threadIdx.x;

    if(th_tile < rowA)
    {
        for(int ii=0; ii<width;ii++)
        {
            for(int i = th_tile * colA + lane_id; i< (th_tile+1) * colA; i+=32)
            {
                double d_matrixA_value = d_matrixA[i];
                double d_dn_value = d_densemtx[ii + (i % colA) * width];
                atomicAdd(&d_resultmtx[th_tile * width  + ii], d_matrixA_value* d_dn_value);               
            }         
        }
    }
}


void Swift_gpu(char *filename,
               float &time_Swift_gpu,
               double &gflops_Swift_gpu,
               int rowA,
               int colA,
               int width,
               int nnz,
               double *h_matrixA,
               double *h_densemtx,
               double *h_resultmtx)
{
    double *d_matrixA;

    double *d_densemtx;
    double *d_resultmtx;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_matrixA, (rowA * colA) * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_densemtx, (colA * width)*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_resultmtx, (rowA * width)*sizeof(double)));


    CHECK_CUDA_ERROR(cudaMemcpy(d_matrixA, h_matrixA, (rowA * colA) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_densemtx, h_densemtx, (colA * width) * sizeof(double), cudaMemcpyHostToDevice));



    int num_threads = TILESIZE *warpperblock;
    int num_blocks = ceil(( double) rowA / (double)warpperblock); 

    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);

    Swift_SpMM_kernel<<<num_blocks, num_threads>>>(rowA, colA, width, nnz,
                                                   d_matrixA,
                                                   d_densemtx,
                                                   d_resultmtx);


    cudaDeviceSynchronize();
    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&time_Swift_gpu, event1, event2);
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_resultmtx, d_resultmtx, rowA * width * sizeof(double), cudaMemcpyDeviceToHost));

  



    CHECK_LAST_CUDA_ERROR();

    cudaFree(d_matrixA);
    cudaFree(d_densemtx);
    cudaFree(d_resultmtx);
}