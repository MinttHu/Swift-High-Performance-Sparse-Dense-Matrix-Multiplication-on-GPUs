#include <cuda_runtime.h>
#include <iostream>

__device__ float warp_prefix_sum(float val) {
    // 遍历 1, 2, 4, 8, 16 步长，逐步累加前面的值
    for (int offset = 1; offset < 32; offset *= 2) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) {  // 仅累加有效数据
            val += temp;
        }
    }
    return val;
}

__global__ void Swift_GPU(int rowA, int colA, 
                          int width, int nnz,
                          int tilenum,
                          int *d_tile_ptr,
                          int *d_tile_colidx,
                          int *d_format,
                          int *d_sortrowindex,
                          int *d_sortrowidx,
                          double *d_sortval,
                          double *d_densemtx,
                          double *d_resultmtx)
{
    
    int tile_colidx = d_tile_colidx[blockIdx.x];
    int tile_start = d_tile_ptr[blockIdx.x];
    const int lane_id = (TILESIZE -1) & threadIdx.x;  
    double d_sp_val = d_sortval[tile_start + lane_id];
    int rowindex = d_sortrowidx[tile_start + lane_id];
    int format = d_format[blockIdx.x];

    switch (format)
    {
        case 0:
        {
            int count =width / 32 ;
            //__shared__ double val_tmp[width];//SC_SIZE：2048
            for(int i = 0; i <count; i++)
            {
                int dense_index = tile_colidx+ lane_id + (threadIdx.x / 32) * colA + i*32*colA;
                int result_index = rowindex + (threadIdx.x / 32) * rowA + i*32*rowA;
                double d_dn_value = d_densemtx[dense_index];
               
                double val = 0;
                val = d_sp_val * d_dn_value;
                val += __shfl_down(val, 16);
                val += __shfl_down(val, 8);
                val += __shfl_down(val, 4);
                val += __shfl_down(val, 2);
                val += __shfl_down(val, 1);
                if((threadIdx.x & 31) == 0)
                {
                    atomicAdd(&d_resultmtx[result_index], val);
                }
            }                
            if(width %32 !=0)
            {
                if((threadIdx.x /32 ) < (width %32))
                {
                    int dense_index = tile_colidx+ lane_id + (threadIdx.x / 32) * colA + (width/32)*32*colA;
                    int result_index = rowindex + (threadIdx.x / 32) * rowA + (width/32)*32*rowA;
                    double d_dn_value = d_densemtx[dense_index];
               
                    double val = 0;
                    val = d_sp_val * d_dn_value;
                    val += __shfl_down(val, 16);
                    val += __shfl_down(val, 8);
                    val += __shfl_down(val, 4);
                    val += __shfl_down(val, 2);
                    val += __shfl_down(val, 1);
                    if((threadIdx.x & 31) == 0)
                    {
                        atomicAdd(&d_resultmtx[result_index], val);
                    }
                }                
            }
        }
        break;

        case 1:
        {
            int count =width / 32 ;
            for(int i = 0; i <count; i++)
            {
                int dense_index = tile_colidx+ lane_id + (threadIdx.x / 32) * colA + i*32*colA;
                int result_index = rowindex + (threadIdx.x / 32) * rowA + i*32*rowA;
                double d_dn_value = d_densemtx[dense_index];
                atomicAdd(&d_resultmtx[result_index], d_sp_val * d_dn_value);
            }
            if ((width % 32) != 0)  
            {
                if((threadIdx.x /32 ) < (width %32))
                {

                    int dense_index = tile_colidx+ lane_id + (threadIdx.x / 32) * colA + (width / 32)*32*colA;
                    int result_index = rowindex + (threadIdx.x / 32) * rowA + (width / 32)*32*rowA;
                    double d_dn_value = d_densemtx[dense_index];
                    atomicAdd(&d_resultmtx[result_index], d_sp_val * d_dn_value);  
                }              
            }

        }
        break;
        case 2:
        {
            //__shared__ double shared_mem[32*width];
            //extern __shared__ double shared_mem[];
            int count =width / 32 ;
            for(int i = 0; i <count; i++)
            {
                
                int dense_index = tile_colidx+ lane_id + (threadIdx.x / 32) * colA + i*32*colA;
                int result_index = rowindex + (threadIdx.x / 32) * rowA + i*32*rowA;
                double d_dn_value = d_densemtx[dense_index];
                atomicAdd(&d_resultmtx[result_index], d_sp_val * d_dn_value);          
                
            }
            if ((width % 32) != 0)  
            {
                if((threadIdx.x /32 ) < (width %32))    
                {

                    int dense_index = tile_colidx+ lane_id + (threadIdx.x / 32) * colA + (width / 32)*32*colA;
                    int result_index = rowindex + (threadIdx.x / 32) * rowA + (width / 32)*32*rowA;
                    double d_dn_value = d_densemtx[dense_index];
                    atomicAdd(&d_resultmtx[result_index], d_sp_val * d_dn_value);   
                }            
            }            
        }
        break;
    }           
}


__global__ void Swift_reside_kernel(int rowA, int colA, 
                                    int width, int nnz,
                                    int reside_col,
                                    int *d_reside_ptr,
                                    int *d_reside_rowidx,
                                    double *d_reside_val,
                                    int d_numblk,
                                    unsigned int *dd_colofblk,
                                    int *dd_blkstart,
                                    int *dd_blkstop,
                                    int *dd_blkptr,
                                    double *d_densemtx,
                                    double *d_resultmtx)
{
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    const int th_tile = global_id >> 5;

    const int local_warp_id = threadIdx.x >> 5;

    const int lane_id = threadIdx.x % 32;


    if(th_tile < d_numblk)
    {
        int colidx = dd_colofblk[th_tile];
        int signbit = (colidx >> 31) & 0x1;

        int colidx_1 = signbit == 1? colidx & 0x7FFFFFFF : colidx;
        int real_colidx = colA - colidx_1 - 1;
        int start = signbit == 1 ? dd_blkstart[th_tile] : d_reside_ptr[colidx_1];
        int stop = signbit ==1 ? dd_blkstop[th_tile] : d_reside_ptr[colidx_1 + 1];   
        for(int i = start + lane_id; i< stop; i+=32)
        {
            int drowidx = d_reside_rowidx[i];
            double d_val = d_reside_val[i];
            for(int j = 0; j< width;j++)
            {

                double d_dn_value = d_densemtx[real_colidx + colA * j];
                atomicAdd(&d_resultmtx[drowidx + rowA * j], d_val * d_dn_value);
            }
        }    
    }
}


void Swift_GPU(char *filename,
               float &time_fastload_gpu,
               double &gflops_fastload_gpu,
               slide_matrix *matrix,
               int rowA,
               int colA,
               int width,
               int nnz,
               double *h_densemtx,
               double *h_resultmtx,
               double *h_golden_resultmtx)
{
    int tilenum = matrix->tilenum;
    int regular_nnz = matrix->regular_nnz;

    int *tile_ptr = matrix->tile_ptr;
    int *tile_colidx = matrix->tile_colidx;
    int *tile_format = matrix->tile_format;
    int *sortrowidx = matrix->sortrowidx;
    int *sortrowindex = matrix->sortrowindex;
    double *sortval = matrix->sortval;



    int *d_tile_ptr;
    int *d_tile_colidx;
    int *d_format;
    int *d_sortrowidx;
    int *d_sortrowindex;
    double *d_sortval;


    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_ptr, (tilenum + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_colidx, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_format, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowidx,(regular_nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowindex,(regular_nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortval,(regular_nnz)*sizeof(double)));



    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_ptr, tile_ptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_colidx, tile_colidx, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_format, tile_format, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowidx, sortrowidx, (regular_nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowindex, sortrowindex, (regular_nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortval, sortval, (regular_nnz) * sizeof(double), cudaMemcpyHostToDevice));


//reside part
    int h_reside_col = matrix->reside_col;
    int h_reside_nnz = matrix->reside_nnz;
    int *h_reside_ptr = matrix->reside_cscptr;
    int *h_reside_rowidx = matrix->reside_cscrowidx;
    double *h_reside_val = matrix->reside_val;

    int d_numblk = matrix->numblk;
    unsigned int *colofblk_h = matrix->colorblk;
    int *blkstart_h = matrix->blkstart;
    int *blkstop_h = matrix->blkstop;
    int *blkptr_h = matrix->blkptr;   


    int *d_reside_rowidx;
    double *d_reside_val;
    int *d_reside_ptr;

    unsigned int *d_colofblk; //
    int *d_blkstart;          //
    int *d_blkstop;           //
    int *d_blkptr;            //


    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reside_ptr, (h_reside_col + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reside_rowidx, (h_reside_nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reside_val, (h_reside_nnz)*sizeof(double)));

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_colofblk, (d_numblk) * sizeof(unsigned int)));//
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_blkstart, (d_numblk) * sizeof(int)));//
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_blkstop, (d_numblk) * sizeof(int)));//
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_blkptr, (d_numblk) * sizeof(int)));//

    CHECK_CUDA_ERROR(cudaMemcpy(d_reside_ptr, h_reside_ptr, (h_reside_col + 1 ) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_reside_val, h_reside_val, (h_reside_nnz) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_reside_rowidx, h_reside_rowidx, (h_reside_nnz) * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemcpy(d_colofblk, colofblk_h, (d_numblk) * sizeof(unsigned int), cudaMemcpyHostToDevice)); //
    CHECK_CUDA_ERROR(cudaMemcpy(d_blkstart, blkstart_h, (d_numblk) * sizeof(int), cudaMemcpyHostToDevice)); //
    CHECK_CUDA_ERROR(cudaMemcpy(d_blkstop, blkstop_h, (d_numblk) * sizeof(int), cudaMemcpyHostToDevice)); //
    CHECK_CUDA_ERROR(cudaMemcpy(d_blkptr, blkptr_h, (d_numblk) * sizeof(int), cudaMemcpyHostToDevice)); //

    double *d_densemtx;
    double *d_resultmtx;
    cudaMalloc((void **)&d_densemtx, colA * width  * sizeof(double));
    cudaMalloc((void **)&d_resultmtx, rowA * width * sizeof(double));
    cudaMemset(d_resultmtx, 0, rowA * width*sizeof(double));

    cudaMemcpy(d_densemtx, h_densemtx, colA *  width * sizeof(double), cudaMemcpyHostToDevice);

    int num_threads = TILESIZE *warpperblock;
    int num_blocks = ceil(( double)tilenum / (double)warpperblock); 

    int num_thread_reside = TILESIZE * warpperblock;
    int num_blocks_reside = ceil((float)d_numblk / (float)warpperblock);


//    cudaStream_t stream1, stream2;
 //   cudaStreamCreate(&stream1);
 //   cudaStreamCreate(&stream2);    



    if(tilenum !=0 && h_reside_nnz !=0)
    {
        cudaEvent_t event1,event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);

        cudaEventRecord(event1,0);
 
        //Swift_GPU<<<tilenum, 1024,32*width*sizeof(double)>>>(rowA, colA,
        Swift_GPU<<<tilenum, 1024,0>>>(rowA, colA,  
                                               width, nnz,
                                               tilenum,
                                               d_tile_ptr,
                                               d_tile_colidx,
                                               d_format,
                                               d_sortrowindex,
                                               d_sortrowidx,
                                               d_sortval,
                                               d_densemtx,
                                               d_resultmtx);

        Swift_reside_kernel<<<num_blocks_reside, num_thread_reside>>>(rowA, colA, 
                                                                      width, nnz,
                                                                      h_reside_col,
                                                                      d_reside_ptr,
                                                                      d_reside_rowidx,
                                                                      d_reside_val,
                                                                      d_numblk,
                                                                      d_colofblk,
                                                                      d_blkstart,
                                                                      d_blkstop,
                                                                      d_blkptr,
                                                                      d_densemtx,
                                                                      d_resultmtx);

       cudaEventRecord(event2,0);


        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&time_fastload_gpu, event1, event2);
        cudaDeviceSynchronize();
    }
    else if(tilenum !=0 && h_reside_nnz ==0)
    {
        cudaEvent_t event1,event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);

        cudaEventRecord(event1,0);
  
        //Swift_GPU<<<tilenum, 1024,32*width*sizeof(double)>>>(rowA, colA, 
        Swift_GPU<<<tilenum, 1024,0>>>(rowA, colA,
                                               width, nnz,
                                               tilenum,
                                               d_tile_ptr,
                                               d_tile_colidx,
                                               d_format,
                                               d_sortrowindex,
                                               d_sortrowidx,
                                               d_sortval,
                                               d_densemtx,
                                               d_resultmtx);
                                                          
        cudaEventRecord(event2,0);

        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&time_fastload_gpu, event1, event2);
        cudaDeviceSynchronize();
    }
    else if(tilenum ==0 && h_reside_nnz !=0)
    {
       cudaEvent_t event1,event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1,0);         
        Swift_reside_kernel<<<num_blocks_reside, num_thread_reside>>>(rowA, colA, 
                                                                      width, nnz,
                                                                      h_reside_col,
                                                                      d_reside_ptr,
                                                                      d_reside_rowidx,
                                                                      d_reside_val,
                                                                      d_numblk,
                                                                      d_colofblk,
                                                                      d_blkstart,
                                                                      d_blkstop,
                                                                      d_blkptr,
                                                                      d_densemtx,
                                                                      d_resultmtx);
        cudaEventRecord(event2,0);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&time_fastload_gpu, event1, event2);
        cudaDeviceSynchronize();
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_resultmtx, d_resultmtx, rowA * width * sizeof(double), cudaMemcpyDeviceToHost));

    double *h_resultmtx1 = (double *)malloc(sizeof(double) * rowA*width);
    memset(h_resultmtx1,0, sizeof(double) * rowA*width);
    dense_mtx2dense_mtx_spmm(width, rowA, h_resultmtx, h_resultmtx1);
    int error_FastLoad_spmm = 0;
    ResultVerify(h_resultmtx1, h_golden_resultmtx, rowA * width , error_FastLoad_spmm);

    if (error_FastLoad_spmm != 0)
    {  
        printf("Swift GPU SpMM (dense mtx col-major) Check NO PASS! error = %d \n", error_FastLoad_spmm);
        time_fastload_gpu = -1;
    }

 

    CHECK_LAST_CUDA_ERROR();

    cudaFree(d_tile_ptr);
    cudaFree(d_tile_colidx);
    cudaFree(d_format);
    cudaFree(d_sortrowidx);
    cudaFree(d_sortval);
    cudaFree(d_densemtx);
    cudaFree(d_resultmtx);
}