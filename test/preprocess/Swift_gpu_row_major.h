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

__global__ void Swift_SpMM_kernel3(int rowA, int colA, int width, int nnz,
                                    int tilenum,
                                    int *d_tile_ptr,
                                    //int *d_tile_tall,
                                    int *d_tile_len,
                                    int *d_tile_colidx,
                                    int *d_format,
                                    //int *d_countrow,
                                    //int *d_segmentoffset,
                                    int *d_sortrowindex,
                                    int *d_sortrowidx,
                                    double *d_sortval,
                                    double *d_densemtx,
                                    double *d_resultmtx)
{
    int global_id = blockIdx.x *blockDim.x + threadIdx.x;
    const int th_tile = global_id >> 5;

    __shared__ double s_x[warpperblock *slidesize];
    __shared__ double s_y[warpperblock *slidesize];
    __shared__ double s_s[warpperblock *slidesize];

    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (TILESIZE -1) & threadIdx.x;
    //const int lane_id = threadIdx.x %32;
    double *s_x_warp = &s_x[local_warp_id *slidesize];
    double *s_y_warp = &s_y[local_warp_id *slidesize];
    double *s_s_warp = &s_s[local_warp_id *slidesize];

    if(th_tile < tilenum)
    {
        int tilelen = d_tile_len[th_tile];
        int tile_start = d_tile_ptr[th_tile];
        int tile_stop = d_tile_ptr[th_tile+1];
        //int tile_tall = d_tile_tall[th_tile];
        int tile_colidx = d_tile_colidx[th_tile];

        int j = tile_start + lane_id ;
        int d_rowidx = d_sortrowidx[j];
        //printf("th_tile :%d tilelen:%d tile_start: %d \n", th_tile,tilelen,tile_start);
        if(lane_id <tilelen)
        {
            for(int ii=0; ii<width;ii++)
            {
                double d_sp_value = d_sortval[j];
                double d_dn_value = d_densemtx[(tile_colidx+lane_id) * width + ii];
                //printf("sp:%f, rowidx: %d coidx: %d dn: %f,k %d\n",d_sp_value, d_rowidx, tile_colidx+lane_id,d_dn_value, ii);                
                atomicAdd(&d_resultmtx[d_rowidx * width + ii], d_sp_value * d_dn_value);             
            }         

        }
    }
}


__global__ void Swift_SpMM_reside_kernel3(int rowA, int colA, int width, int nnz,
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
    //const int lane_id = (TILESIZE -1) & threadIdx.x;
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
                double d_dn_value = d_densemtx[real_colidx * width +j];
                atomicAdd(&d_resultmtx[drowidx * width + j], d_val * d_dn_value);
            }


        }    


    }

}


void Swift_gpu2(char *filename,
                float &time_Swift_gpu,
                double &gflops_Swift_gpu,
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
    int *tile_ptr = matrix->tile_ptr;
    //int *tile_tall = matrix->tile_tall;
    int *tile_len = matrix->tile_len;
    int *tile_colidx = matrix->tile_colidx;
    int *tile_format = matrix->tile_format;
    int *sortrowidx = matrix->sortrowidx;
    double *sortval = matrix->sortval;


    int regular_nnz = matrix->regular_nnz;


    int segsum = matrix->segsum;
    int *sortrowindex = matrix->sortrowindex;


    int *d_tile_ptr;
    //int *d_tile_tall;
    int *d_tile_len;
    int *d_tile_colidx;
    int *d_format;
    int *d_sortrowidx;
    double *d_sortval;

//    int *d_countrow;
//    int *d_segmentoffset;
    int *d_sortrowindex;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_ptr, (tilenum + 1) * sizeof(int)));
    //CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_tall, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_len, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_tile_colidx, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_format, (tilenum)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowidx,(regular_nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortval,(regular_nnz)*sizeof(double)));

//    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_countrow,(tilenum+1)*sizeof(int)));
//    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_segmentoffset,(segsum+1)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortrowindex,(regular_nnz)*sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_ptr, tile_ptr, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice));
    //CHECK_CUDA_ERROR(cudaMemcpy(d_tile_tall, tile_tall, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_len, tile_len, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tile_colidx, tile_colidx, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
   // CHECK_CUDA_ERROR(cudaMemcpy(d_format, tile_format, (tilenum) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowidx, sortrowidx, (regular_nnz) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sortval, sortval, (regular_nnz) * sizeof(double), cudaMemcpyHostToDevice));

//    CHECK_CUDA_ERROR(cudaMemcpy(d_countrow, countrow, (tilenum+1) * sizeof(int), cudaMemcpyHostToDevice));
//    CHECK_CUDA_ERROR(cudaMemcpy(d_segmentoffset, segmentoffset, (segsum+1) * sizeof(int), cudaMemcpyHostToDevice));
    //CHECK_CUDA_ERROR(cudaMemcpy(d_sortrowindex, sortrowindex, (regular_nnz) * sizeof(int), cudaMemcpyHostToDevice));


//reside part
    int h_reside_col = matrix->reside_col;
    int h_reside_nnz = matrix->reside_nnz;
    int *h_reside_ptr = matrix->reside_cscptr;
    int *h_reside_rowidx = matrix->reside_cscrowidx;
    double *h_reside_val = matrix->reside_val;

    int *d_reside_rowidx;
    double *d_reside_val;
    int *d_reside_ptr;

    int d_numblk = matrix->numblk;
    unsigned int *colofblk_h = matrix->colorblk;
    int *blkstart_h = matrix->blkstart;
    int *blkstop_h = matrix->blkstop;
    int *blkptr_h = matrix->blkptr;   


    unsigned int *d_colofblk; //
    int *d_blkstart;          //
    int *d_blkstop;           //
    int *d_blkptr;            //

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_colofblk, (d_numblk) * sizeof(unsigned int)));//
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_blkstart, (d_numblk) * sizeof(int)));//
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_blkstop, (d_numblk) * sizeof(int)));//
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_blkptr, (d_numblk) * sizeof(int)));//

    CHECK_CUDA_ERROR(cudaMemcpy(d_colofblk, colofblk_h, (d_numblk) * sizeof(unsigned int), cudaMemcpyHostToDevice)); //
    CHECK_CUDA_ERROR(cudaMemcpy(d_blkstart, blkstart_h, (d_numblk) * sizeof(int), cudaMemcpyHostToDevice)); //
    CHECK_CUDA_ERROR(cudaMemcpy(d_blkstop, blkstop_h, (d_numblk) * sizeof(int), cudaMemcpyHostToDevice)); //
    CHECK_CUDA_ERROR(cudaMemcpy(d_blkptr, blkptr_h, (d_numblk) * sizeof(int), cudaMemcpyHostToDevice)); //


    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reside_ptr, (h_reside_col + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reside_rowidx, (h_reside_nnz)*sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_reside_val, (h_reside_nnz)*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_reside_ptr, h_reside_ptr, (h_reside_col + 1 ) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_reside_val, h_reside_val, (h_reside_nnz) * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_reside_rowidx, h_reside_rowidx, (h_reside_nnz) * sizeof(int), cudaMemcpyHostToDevice));

    double *d_densemtx;
    double *d_resultmtx;
    cudaMalloc((void **)&d_densemtx, colA * width  * sizeof(double));
    cudaMalloc((void **)&d_resultmtx, rowA * width * sizeof(double));
    cudaMemset(d_resultmtx, 0, rowA * width*sizeof(double));

    cudaMemcpy(d_densemtx, h_densemtx, colA *  width * sizeof(double), cudaMemcpyHostToDevice);

    int num_threads = TILESIZE *warpperblock;
    int num_blocks = ceil(( double)tilenum / (double)warpperblock); 

    int num_thread_reside = TILESIZE * warpperblock;
    int num_blocks_reside = ceil((double)d_numblk / (double)warpperblock);

if(tilenum !=0 && h_reside_nnz !=0)
{
    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    Swift_SpMM_kernel3<<<num_blocks, num_threads>>>(rowA, colA, width, nnz,
                                                       tilenum,
                                                       d_tile_ptr,
                                                      //int *d_tile_tall,
                                                       d_tile_len,
                                                       d_tile_colidx,
                                                       d_format,
                                                        //int *d_countrow,
                                                      //int *d_segmentoffset,
                                                       d_sortrowindex,
                                                       d_sortrowidx,
                                                       d_sortval,
                                                       d_densemtx,
                                                       d_resultmtx);
    

    Swift_SpMM_reside_kernel3<<<num_blocks_reside, num_thread_reside>>>(rowA, colA, width, nnz,
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
    

    //cudaDeviceSynchronize();
    cudaEventRecord(event2,0);

    //cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&time_Swift_gpu, event1, event2);
    cudaDeviceSynchronize();
}
else if(tilenum !=0 && h_reside_nnz ==0)
{
        cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //cudaDeviceSynchronize();
    cudaEventRecord(event1,0);
    Swift_SpMM_kernel3<<<num_blocks, num_threads>>>(rowA, colA, width, nnz,
                                                       tilenum,
                                                       d_tile_ptr,
                                                      //int *d_tile_tall,
                                                       d_tile_len,
                                                       d_tile_colidx,
                                                       d_format,
                                                        //int *d_countrow,
                                                      //int *d_segmentoffset,
                                                       d_sortrowindex,
                                                       d_sortrowidx,
                                                       d_sortval,
                                                       d_densemtx,
                                                       d_resultmtx);
    
    
    //cudaDeviceSynchronize();
    cudaEventRecord(event2,0);

    //cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&time_Swift_gpu, event1, event2);
    cudaDeviceSynchronize();
}
else if(tilenum ==0 && h_reside_nnz !=0)
{
    cudaEvent_t event1,event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    //cudaDeviceSynchronize();
    cudaEventRecord(event1,0);    
    Swift_SpMM_reside_kernel3<<<num_blocks_reside, num_thread_reside>>>(rowA, colA, width, nnz,
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
    //cudaDeviceSynchronize();
    cudaEventRecord(event2,0);

    //cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);
    cudaEventElapsedTime(&time_Swift_gpu, event1, event2);
    cudaDeviceSynchronize();
}

    CHECK_CUDA_ERROR(cudaMemcpy(h_resultmtx, d_resultmtx, rowA * width * sizeof(double), cudaMemcpyDeviceToHost));

    int error_Swift_spmm = 0;
    ResultVerify(h_resultmtx, h_golden_resultmtx, rowA * width , error_Swift_spmm);

    if (error_Swift_spmm != 0)
    {
        printf("Swift GPU SpMM (row major) Check NO PASS! error = %d \n", error_Swift_spmm);
        time_Swift_gpu = -1;    
        
    }

    CHECK_LAST_CUDA_ERROR();

    cudaFree(d_tile_ptr);
    //cudaFree(d_tile_tall);
    cudaFree(d_tile_len);
    cudaFree(d_tile_colidx);
    cudaFree(d_format);
    cudaFree(d_sortrowidx);
    cudaFree(d_sortval);
    cudaFree(d_densemtx);
    cudaFree(d_resultmtx);
}