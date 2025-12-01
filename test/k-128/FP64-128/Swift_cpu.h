
void FastLoad_cpu(double &time_fastload_cpu,
                  slide_matrix *matrix,
                  int nnz,
                  int rowA,
                  int colA,
                  int width_dense_mtx,
               
                  double *dense_mtx,
                  double *result_mtx)

{

    double *sortval = matrix->sortval;
    int *sortrowidx = matrix->sortrowidx;
    int *tile_ptr = matrix->tile_ptr;
    int *tile_len = matrix->tile_len;
    int *tile_colidx = matrix->tile_colidx;
    int tilenum = matrix->tilenum;


    int *reside_cscptr = matrix->reside_cscptr;
    int *reside_cscrowidx = matrix->reside_cscrowidx;
    double *reside_val = matrix->reside_val;   
    int reside_col = matrix->reside_col; 

timeval t1, t2;
gettimeofday(&t1, NULL);


    for(int i=0;i<tilenum;i++)
    {
        int start = tile_ptr[i];
        int stop = tile_ptr[i+1];
        for(int j = start; j<stop;j++)
        {
            int len=tile_len[i];
            int colidx=tile_colidx[i] + (j-start)%len ;
            int rowidx = sortrowidx[j];
            //printf("len: %d \n", len);
            for(int k=0 ; k<width_dense_mtx;k++)
            {
                
                //printf("sp:%f, dense: %f k:%d target:%d ",sortval[j], dense_mtx[colidx *width_dense_mtx + k], k,rowidx *width_dense_mtx + k);
                result_mtx[rowidx *width_dense_mtx + k] += sortval[j] * dense_mtx[colidx * width_dense_mtx +k];
            }
            //printf("\n");
            
        }
    }

    for(int i=0; i<reside_col;i++)
    {
        int colidx = colA - i - 1;
        for(int j = reside_cscptr[i]; j< reside_cscptr[i+1]; j++)
        {
            int rowidx = reside_cscrowidx[j];
            
            for(int k = 0; k < width_dense_mtx; k++)
            {
                result_mtx[rowidx * width_dense_mtx +k] += reside_val[j] * dense_mtx[colidx * width_dense_mtx + k];

            }
        }
    }
gettimeofday(&t2, NULL); 
time_fastload_cpu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    /*
    for(int i=0; i< rowA; i++)
    {
      for(int j = 0 ; j<width_dense_mtx;j++)
      {
        printf("%f ", result_mtx[i*width_dense_mtx+j]);
      }
      printf("\n");
    }
    */

}