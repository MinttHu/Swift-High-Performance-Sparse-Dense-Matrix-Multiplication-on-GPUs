#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>



__global__ void Pre4sortedx(int colA,int width, int *sortedIndices, double *densemtx, double *sort_densemtx)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < colA)
    {
        int index = sortedIndices[tid];
        //double x_tmp = densemtx[index];
        //sort_densemtx[tid] = x_tmp;     
        for(int i=0 ; i<width ; i++)
        {
            sort_densemtx[tid * width + i] = densemtx[index * width + i];
            //printf("tid:%d index %d value:%f \n", tid,index, densemtx[index]);  
        }
    }   

}

__global__ void Pre4SortIdxVal(int colA, int *sortedIndices,int *sortedcol, int *cscColPtr, int *cscRowIdx, double *cscVal, int *sortedrowIdxTmp, double *sortedValTmp)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < colA)
    {
        int index = sortedIndices[tid];
        int start = cscColPtr[index];
        int stop = cscColPtr[index+1];
        int tmp = 0;
        for(int i = 0; i<tid; i++)
        {
            tmp = tmp +sortedcol[i];
        }
        for(int j = start; j<stop; j++)
        {
            int rowidxTmp = cscRowIdx[j];
            sortedrowIdxTmp[tmp] = rowidxTmp;
            double valTmp = cscVal[j];
            sortedValTmp[tmp] = valTmp;
            tmp = tmp+1;
        }
    }    
}


void ColSort(double &timeForSort,
             int colA,
             int nnz,
             int width,
             int *nnzPerCol,
             int *cscColPtr,
             int *cscRowIdx,
             double *cscVal, 

             int *sortrowidxTmp,
             double *sortvalTmp,
             int *sortnnzTmp, 

             double *densemtx,
             double *sort_densemtx)
{
    double *d_densemtx;
    double *d_sort_densemtx;
    
    int *d_cscColPtr;
    int *d_cscRowIdx;
    double *d_cscVal;

    int *d_sortedRowIdxTmp;
    double *d_sortedValTmp;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_densemtx, (colA * width) * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sort_densemtx, (colA * width)*sizeof(double))); 
    cudaMemset(d_sort_densemtx, 0, colA * width*sizeof(double));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cscColPtr, (colA+1)*sizeof(int)));  
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cscRowIdx, (nnz)*sizeof(int))); 
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cscVal, (nnz)*sizeof(double))); 

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortedRowIdxTmp, (nnz)*sizeof(int))); 
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortedValTmp, (nnz)*sizeof(double))); 

    CHECK_CUDA_ERROR(cudaMemcpy(d_densemtx, densemtx, (colA * width) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cscColPtr, cscColPtr, (colA+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cscRowIdx, cscRowIdx, (nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cscVal, cscVal, (nnz) * sizeof(double), cudaMemcpyHostToDevice));

    //timeForSort=0;
    //timeval tsortC1, tsortC2;
    //gettimeofday(&tsortC1, NULL);    

    thrust::host_vector<int> h_data(colA);
    for (int i = 0; i < colA; ++i) 
    {
        h_data[i] = nnzPerCol[i];
    }
  
    thrust::device_vector<int> d_nnzPerCol = h_data;
    thrust::device_vector<int> d_indices(colA);
    thrust::sequence(d_indices.begin(), d_indices.end());

    timeForSort=0;
    timeval tsortC1, tsortC2;
    gettimeofday(&tsortC1, NULL);  

    // 对数组和索引进行排序
    thrust::sort_by_key(d_nnzPerCol.begin(), d_nnzPerCol.end(), d_indices.begin()); 

    int* d_data_indice = thrust::raw_pointer_cast(d_indices.data());
    int* d_data_sortedCol = thrust::raw_pointer_cast(d_nnzPerCol.data());

    const int blockSize=256;
    const int numBlocksColSort = (colA+ blockSize -1 ) / blockSize;
    Pre4sortedx<<<numBlocksColSort, blockSize>>>(colA,width, d_data_indice, d_densemtx, d_sort_densemtx);

    Pre4SortIdxVal<<<numBlocksColSort, blockSize>>>(colA, d_data_indice, d_data_sortedCol, d_cscColPtr, d_cscRowIdx, d_cscVal, d_sortedRowIdxTmp, d_sortedValTmp);

    gettimeofday(&tsortC2, NULL);
    timeForSort= (tsortC2.tv_sec - tsortC1.tv_sec) * 1000.0 + (tsortC2.tv_usec - tsortC1.tv_usec) / 1000.0;  
    thrust::host_vector<int> h_sortedNnzPerCol = d_nnzPerCol;

    //gettimeofday(&tsortC2, NULL);
    //timeForSort= (tsortC2.tv_sec - tsortC1.tv_sec) * 1000.0 + (tsortC2.tv_usec - tsortC1.tv_usec) / 1000.0;    

    CHECK_CUDA_ERROR(cudaMemcpy(sortrowidxTmp, d_sortedRowIdxTmp,(nnz) * sizeof(int), cudaMemcpyDeviceToHost));  
    CHECK_CUDA_ERROR(cudaMemcpy(sortvalTmp, d_sortedValTmp,(nnz) * sizeof(double), cudaMemcpyDeviceToHost)); 
    CHECK_CUDA_ERROR(cudaMemcpy(sort_densemtx, d_sort_densemtx,(colA * width) * sizeof(double), cudaMemcpyDeviceToHost));       
 
    for (int i = 0; i < colA; ++i) 
    {
       sortnnzTmp[i] = h_sortedNnzPerCol[i];    
    }  



    CHECK_CUDA_ERROR(cudaFree(d_cscColPtr));
    CHECK_CUDA_ERROR(cudaFree(d_cscRowIdx));    
    CHECK_CUDA_ERROR(cudaFree(d_cscVal));  
    CHECK_CUDA_ERROR(cudaFree(d_sortedRowIdxTmp));  
    CHECK_CUDA_ERROR(cudaFree(d_sortedValTmp));  

    CHECK_CUDA_ERROR(cudaFree(d_densemtx));
    CHECK_CUDA_ERROR(cudaFree(d_sort_densemtx));

}