#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>




struct KV {
    int key;
    int orig_idx;
};

// block-level bitonic sort for 256 elements; each block has 256 threads
// Uses shared memory

__device__ inline void bitonic_sort_256(KV *sdata, int tid) {
    // standard bitonic in-place
    // sized for 256 (2^8)
    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                // ascending or descending depends on bit of tid/k
                if ((tid & k) == 0) {
                    // ascending phase
                    if (sdata[tid].key > sdata[ixj].key) {
                        // swap
                        KV tmp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = tmp;
                    }
                } else {
                    // descending phase
                    if (sdata[tid].key < sdata[ixj].key) {
                        KV tmp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
}


__global__ void sort_tile_groups_kernel(const int *d_sortrowidx,
                                        const int *d_tile_ptr,
                                        int *d_back_rowidx,
                                        int *d_val_position,
                                        int loop)
{
    int group_id = blockIdx.x;
    if (group_id >= loop) return;

    int start_tile = group_id * 8;
    int base = d_tile_ptr[start_tile]; 


    extern __shared__ KV sdata[]; 

    int tid = threadIdx.x;
    int global_idx = base + tid;


    sdata[tid].key = d_sortrowidx[global_idx];
    sdata[tid].orig_idx = tid; 
    __syncthreads();

    bitonic_sort_256(sdata, tid);
    __syncthreads();


    int write_pos = base + tid;
    d_back_rowidx[write_pos] = sdata[tid].key;
    int orig_local_idx = sdata[tid].orig_idx;
    d_val_position[base + orig_local_idx] = tid;
}

__global__ void sort_remainder_kernel(const int *d_sortrowidx,
                                      const int *d_tile_ptr,
                                      int *d_back_rowidx,
                                      int *d_val_position,
                                      int start_tile,
                                      int stop_tile,
                                      int numEle)
{
    int tid = threadIdx.x;
    extern __shared__ KV sdata_rem[]; 

    int base = d_tile_ptr[start_tile];

    if (tid < numEle) {
        sdata_rem[tid].key = d_sortrowidx[base + tid];
        sdata_rem[tid].orig_idx = tid;
    } else {
        sdata_rem[tid].key = INT_MAX;
        sdata_rem[tid].orig_idx = -1;
    }
    __syncthreads();

    bitonic_sort_256(sdata_rem, tid);
    __syncthreads();

    if (tid < numEle) {
        d_back_rowidx[base + tid] = sdata_rem[tid].key;
        int orig_local_idx = sdata_rem[tid].orig_idx;
        if (orig_local_idx >= 0)
            d_val_position[base + orig_local_idx] = tid;
    }
}




void row_idx_sort_gpu(slide_matrix* matrix, int *tmp_sortrowidx, float &ms)
{
    int tilenum = matrix->tilenum;
    int nnz_regu = matrix->regular_nnz;

    int* d_sortrowidx;
    int* d_val_position;
    int* d_tile_ptr;
    int* d_back_rowidx;
    cudaMalloc(&d_sortrowidx, nnz_regu * sizeof(int));
    cudaMalloc(&d_val_position, nnz_regu * sizeof(int));
    cudaMalloc(&d_back_rowidx, nnz_regu * sizeof(int));    
    cudaMalloc(&d_tile_ptr, (tilenum + 1) * sizeof(int));

    cudaMemcpy(d_sortrowidx, tmp_sortrowidx, nnz_regu * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_ptr, matrix->tile_ptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_val_position, 0, nnz_regu * sizeof(int));
    cudaMemset(d_back_rowidx, 0, nnz_regu * sizeof(int));

    int *h_val_position = (int *)malloc(nnz_regu * sizeof(int));
    memset(h_val_position, 0, nnz_regu * sizeof(int));    


    int *h_rowidx = (int *)malloc(nnz_regu * sizeof(int));
    memset(h_rowidx, 0, nnz_regu * sizeof(int));      
    

    int loop = tilenum / 8;

    // timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (loop > 0)
    {
        dim3 grid(loop);
        dim3 block(256);
        size_t shared_bytes = 256 * sizeof(KV); // shared mem per block
        sort_tile_groups_kernel<<<grid, block, shared_bytes>>>(d_sortrowidx, d_tile_ptr, d_back_rowidx, d_val_position, loop);
        cudaGetLastError();
    }

    // remainder
    if (tilenum % 8 != 0)
    {
        int start_tile = (tilenum / 8) * 8;
        int stop_tile = tilenum;
        int numEle = (stop_tile - start_tile) * 32;
        // launch single block for remainder
        dim3 grid2(1);
        dim3 block2(256);
        size_t shared_bytes2 = 256 * sizeof(KV);
        sort_remainder_kernel<<<grid2, block2, shared_bytes2>>>(d_sortrowidx, d_tile_ptr, d_back_rowidx, d_val_position, start_tile, stop_tile, numEle);
        cudaGetLastError();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
 


    cudaMemcpy(h_rowidx, d_back_rowidx, nnz_regu * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_val_position, d_val_position, nnz_regu * sizeof(int), cudaMemcpyDeviceToHost);



    int *val_position = matrix->val_position;
    int *g_rowidx = matrix->sortrowidx;
    int error1 = 0;
    int error2 = 0;
    for(int i=0; i<nnz_regu; i++)
    {
        if(h_rowidx[i] != g_rowidx[i])
        {
            error1++;
        }
        if (h_val_position[i] != val_position[i])
        {
            error2++;
        }     
    }
    if(error1 != 0)
    {
        printf("error rowidx: %d\n", error1);
        ms = -1;
    }
    else
    {
        printf("rowidx sort correct! time: %f\n", ms);
    }
    /*
    if(error2 != 0)
    {
        printf("error val_position: %d\n", error2);
    }
        */



    cudaFree(d_sortrowidx);
    cudaFree(d_val_position);
    cudaFree(d_tile_ptr);
}