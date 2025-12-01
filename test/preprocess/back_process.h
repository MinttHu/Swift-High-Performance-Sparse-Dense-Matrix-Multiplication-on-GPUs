#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "omp.h"
#include <algorithm>
#include <vector>


#include <iostream>
#include <algorithm> 

struct IndexedValue {
    int value;
    int original_index;
};

bool compare(IndexedValue a, IndexedValue b) {
    return a.value < b.value;
}


struct KV {
    int key;
    int orig_idx;
};


__device__ inline void bitonic_sort_256(KV *sdata, int tid) {
    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
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


void row_idx_sort(slide_matrix *matrix, double *dense_mtx,int rowA, int colA, int width, float &time1, float &time2)
{
    int tilenum = matrix->tilenum;
    int *tile_ptr = matrix->tile_ptr;
    int *sortrowidx = matrix->sortrowidx;
    int nnz_regu = matrix->regular_nnz;

    matrix->val_position = (int *)malloc(nnz_regu * sizeof(int));
    memset(matrix->val_position, 0, nnz_regu * sizeof(int));
    int *val_position = matrix->val_position;

    matrix->val_offset = (int *)malloc(nnz_regu * sizeof(int));
    memset(matrix->val_offset, 0, nnz_regu * sizeof(int));
    int *val_offset = matrix->val_offset;


    int *temprowidx = (int *)malloc(256 * sizeof(int));
    memset(temprowidx, 0, 256 * sizeof(int));

    int *back_rowidx = (int *)malloc(nnz_regu * sizeof(int));
    memset(back_rowidx, 0, nnz_regu * sizeof(int));

    int *h_back_rowidx = (int *)malloc(nnz_regu * sizeof(int));
    memset(h_back_rowidx, 0, nnz_regu * sizeof(int));

    int *h_val_position = (int *)malloc(nnz_regu * sizeof(int));
    memset(h_val_position, 0, nnz_regu * sizeof(int));   

    int* d_sortrowidx;
    int* d_val_position;
    int* d_tile_ptr;
    int* d_back_rowidx;
    cudaMalloc(&d_sortrowidx, nnz_regu * sizeof(int));
    cudaMalloc(&d_val_position, nnz_regu * sizeof(int));
    cudaMalloc(&d_back_rowidx, nnz_regu * sizeof(int));    
    cudaMalloc(&d_tile_ptr, (tilenum + 1) * sizeof(int));

    cudaMemcpy(d_sortrowidx, sortrowidx, nnz_regu * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_ptr, matrix->tile_ptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_val_position, 0, nnz_regu * sizeof(int));
    cudaMemset(d_back_rowidx, 0, nnz_regu * sizeof(int));



    IndexedValue temp[256];

timeval tb1, tb2;
gettimeofday(&tb1, NULL);

    int loop = tilenum / 8;
    for (int i = 0; i < loop; i++)
    {
        int start = i * 8;
        int stop = start + 8;
        int count = 0;
        for (int j = tile_ptr[start]; j < tile_ptr[stop]; j++)
        {
            temprowidx[count] = sortrowidx[j];
            count++;
        }
        if(count != 256)
        {
            printf("error load rowidx to temprowidx: count: %d\n", count);
        }

        for (int i = 0; i < 256; ++i) 
        {
            temp[i].value = temprowidx[i];
            temp[i].original_index = i;
        }

        std::sort(temp, temp + 256, compare);

        for (int i = 0; i < 256; ++i) 
        {
            back_rowidx[tile_ptr[start] + i] = temp[i].value;
            val_position[tile_ptr[start] + temp[i].original_index] = i;
        }

    }

    if(tilenum % 8 != 0)
    {
        int start = (tilenum / 8) * 8;
        int stop = tilenum;
        int numEle = (stop - start) * 32;
        IndexedValue temp1[numEle];

        int *temprowidx1 = (int *)malloc(numEle * sizeof(int));
        memset(temprowidx1, 0, numEle * sizeof(int));

        int count = 0;
        for (int j = tile_ptr[start]; j < tile_ptr[stop]; j++)
        {
            temprowidx1[count] = sortrowidx[j];
            count++;
        }
        if(count != numEle)
        {
            printf("error load rowidx to temprowidx (rest): count: %d\n", count);
        }
        for (int i = 0; i < numEle; ++i) 
        {
            temp1[i].value = temprowidx1[i];
            temp1[i].original_index = i;
        }

        std::sort(temp1, temp1 + numEle, compare);

        for (int i = 0; i < numEle; ++i) 
        {
            back_rowidx[tile_ptr[start] + i] = temp1[i].value;
            val_position[tile_ptr[start] + temp1[i].original_index] = i;
        }
                 
    }

gettimeofday(&tb2, NULL); 
time1 = (tb2.tv_sec - tb1.tv_sec) * 1000.0 + (tb2.tv_usec - tb1.tv_usec) / 1000.0;


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
    time1 = 0.0f;
    cudaEventElapsedTime(&time1, start, stop);
 

    cudaMemcpy(h_back_rowidx, d_back_rowidx, nnz_regu * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val_position, d_val_position, nnz_regu * sizeof(int), cudaMemcpyDeviceToHost);

    int error1 = 0;
    for(int i=0; i<nnz_regu; i++)
    {
        if(h_back_rowidx[i] != back_rowidx[i])
        {
            error1++;
        } 
    }
    if(error1 != 0)
    {
        printf("error rowidx: %d\n", error1);
    }



timeval ttime1, ttime2;
gettimeofday(&ttime1, NULL);

    int regu_block_nnz = (tilenum / 8) * 256;

    for(int i = 0; i< nnz_regu; i++)
    {
        //back_rowidx2[i] = sortrowidx[i];
        sortrowidx[i] = back_rowidx[i];
    }

    for(int i=0 ; i<loop; i++)
    {
        int start = i * 8;
        int stop = start + 8;
        int count = 0;
        int compared_val = sortrowidx[tile_ptr[start]];
        for(int j = tile_ptr[start]+1; j<tile_ptr[stop]; j++)
        {
            int val2 = sortrowidx[j];
            if(compared_val == val2)
            {
                count++;
                val_offset[j-count]++;
            }
            else if(compared_val != val2 && count !=0 && j != tile_ptr[stop]-1)
            {
                compared_val = val2;
                count =0;
                //val_offset[j] = -1;
            }
            else if(compared_val != val2 && count == 0 && j!=tile_ptr[stop]-1)
            {
                compared_val = val2;
                count = 0;
                val_offset[j-1] = -1;
            }
            else if(compared_val != val2 && count == 0 && j == tile_ptr[stop]-1)
            {
                compared_val = val2;
                count = 0;
                val_offset[j-1] = -1;
                val_offset[j] = -1;
            }
            else if(compared_val != val2 && count != 0 && j == tile_ptr[stop]-1)
            {
                compared_val = val2;
                count = 0;
                //val_offset[j-1] = -1;
                val_offset[j] = -1;
            }
   
        }
            
    }

   
    if(tilenum % 8 != 0)
    {
        int start = (tilenum / 8) * 8;
        int stop = tilenum;
        int count = 0;
        int compared_val = sortrowidx[tile_ptr[start]];
        for(int j = tile_ptr[start]+1; j<tile_ptr[stop]; j++)
        {
            int val2 = sortrowidx[j];
            if(compared_val == val2)
            {
                count++;
                val_offset[j-count]++;
            }
            else if(compared_val != val2 && count !=0 && j != tile_ptr[stop]-1)
            {
                compared_val = val2;
                count =0;
                //val_offset[j] = -1;
            }
            else if(compared_val != val2 && count == 0 && j != tile_ptr[stop]-1)
            {
                compared_val = val2;
                count = 0;
                val_offset[j-1] = -1;
            }
            else if(compared_val != val2 && count == 0 && j == tile_ptr[stop]-1)
            {
                compared_val = val2;
                count = 0;
                val_offset[j-1] = -1;
                val_offset[j] = -1;
            }            
        }
    }   
      
gettimeofday(&ttime2, NULL); 
time2 = (ttime2.tv_sec - ttime1.tv_sec) * 1000.0 + (ttime2.tv_usec - ttime1.tv_usec) / 1000.0;


    free(temprowidx);
    free(back_rowidx);

}
