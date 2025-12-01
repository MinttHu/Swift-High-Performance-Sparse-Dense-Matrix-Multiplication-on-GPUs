#include <cuda_runtime.h>
#include <iostream>




__global__ void Swift_SpMM_kernel1(int rowA, int colA, int width, int nnz,
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
                double d_dn_value = d_densemtx[ii * colA + i % colA];
                atomicAdd(&d_resultmtx[ th_tile + ii * rowA], d_matrixA_value* d_dn_value);               
            }         
        }
    }
}
void Swift_gpu1(char *filename,
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
    double *d_matrixA1;

    double *d_densemtx1;
    double *d_resultmtx1;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_matrixA1, (rowA * colA) * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_densemtx1, (colA * width)*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_resultmtx1, (rowA * width)*sizeof(double)));


    CHECK_CUDA_ERROR(cudaMemcpy(d_matrixA1, h_matrixA, (rowA * colA) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_densemtx1, h_densemtx, (colA * width) * sizeof(double), cudaMemcpyHostToDevice));

    int num_threads = TILESIZE *warpperblock;
    int num_blocks = ceil(( double)rowA / (double)warpperblock); 

    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaDeviceSynchronize();
    cudaEventRecord(event1,0);

    Swift_SpMM_kernel1<<<num_blocks, num_threads>>>(rowA, colA, width, nnz,
                                                      d_matrixA1,
                                                      d_densemtx1,
                                                      d_resultmtx1);

    cudaDeviceSynchronize();
    cudaEventRecord(event2,0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&time_Swift_gpu, event1, event2);
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaMemcpy(h_resultmtx, d_resultmtx1, rowA * width * sizeof(double), cudaMemcpyDeviceToHost));


    //auxilary_print_mtx(width,rowA,rowA,h_resultmtx);


    double *h_resultmtx1 = (double *)malloc(sizeof(double) * rowA*width);
    memset(h_resultmtx1,0, sizeof(double) * rowA*width);
    dense_mtx2dense_mtx_spmm(width, rowA, h_resultmtx, h_resultmtx1);





    CHECK_LAST_CUDA_ERROR();

    cudaFree(d_matrixA1);

    cudaFree(d_densemtx1);
    cudaFree(d_resultmtx1);
}