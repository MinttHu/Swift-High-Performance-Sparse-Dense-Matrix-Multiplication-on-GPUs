#include <iostream>
#include <vector>
#include "sys/time.h"
//#include "absl/random/random.h"

#include "coo2csc.h"
#include "Swift_cpu.h"
#include "gpu_csr2csc.h"
#include "gpu_cusparse_spmm.h"
#include "Swift_gpu_row_major.h"
#include "Swift_gpu_col_major.h"
#include "Swift.h"

#include "balance.h"
#include "block_catgory.h"
#include "back_process.h"
//#include "gpu_back_process.h"
#include "ColSort_cuda.h"
#include "formatTransform_cuda.h"

#include "Swift_3.h"
#include "Swift_4.h"
#define BN 32

#ifndef VERIFYCSC
#define VERIFYCSC 1
#endif


int main(int argc, char ** argv)
{
    if(argc <2)
    {
        printf("error order\n");
        return 0;
    }

   int device_id = 0;
    // "Usage: ``./spmv -d 0 mtx A.mtx'' for Ax=y on device 0"
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
    filename = new char[fileName.length() + 1];
    std::strcpy(filename, fileName.c_str());
//|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-

    printf("read success,input matrix %s :(%i,%i) nnz =%i \n", filename , rowA,colA,nnz);
  

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
        //csrval[i] = i % 10 + 1;
    }


    double *golden_matrix_c = (double *)malloc(sizeof(double)*rowA *BN);
    memset(golden_matrix_c,0, sizeof(double) * rowA*BN);  



    for(int i=0; i<rowA;i++)
    {
      for(int j=csrrowptr[i]; j<csrrowptr[i+1]; j++)
      {
        for(int k=0; k<BN; k++)
        {
          int dense_index = csrcolidx[j] *  BN;
          golden_matrix_c[i*BN + k] += csrval[j] * dense_matrix[dense_index + k];
        }
      }
    }



    int *cscrowidx = (int *)malloc(sizeof(int) * (nnz));
    memset(cscrowidx, 0, sizeof(int)*(nnz));


    int *csccolptr = (int *)malloc(sizeof(int) * (colA +1));
    memset(csccolptr, 0, sizeof(int)*(colA+1));

    int *nnzpercol=(int *)malloc(sizeof(int)*(colA));
    memset(nnzpercol, 0 ,sizeof(int)*(colA));


    double *cscval= (double *)malloc(sizeof(double)* (nnz));   
    memset(cscval, 0, sizeof(double)*(nnz));

    float csr2cscTime = 0;
    csr_to_csc(csr2cscTime, rowA, colA, nnz, csrval, csrrowptr, csrcolidx, cscval, cscrowidx, csccolptr, nnzpercol);

    double *testcsc_matrix_c = (double *)malloc(sizeof(double)*rowA *BN);
    memset(testcsc_matrix_c,0, sizeof(double) * rowA*BN); 

    for(int i=0; i<colA; i++)
    {
      for(int j=csccolptr[i]; j<csccolptr[i+1]; j++)
      {
        for(int k=0; k<BN; k++)
        {
          int row_index = cscrowidx[j];
          int dense_index = i * BN;
          testcsc_matrix_c[row_index *BN +k] += cscval[j] * dense_matrix[i *BN+k];
        }
      }
    }


#if VERIFYCSC
    int errorCSC=0;
    ResultVerify(testcsc_matrix_c, golden_matrix_c, rowA*BN, errorCSC);
    if(errorCSC !=0)
    {
        printf("error format csc, error = %d\n", errorCSC); 
    }
#endif


    slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));

    int *sortrowidx_tmp = (int *)malloc(sizeof(int)*nnz);

    double *sortval_tmp = (double *)malloc(sizeof(double)*nnz);

    int *sortnnz_tmp= (int *)malloc(sizeof(int)*(colA+1));

    double *sort_dense_mtx = (double *)malloc(sizeof(double)*colA * BN);  
    memset(sort_dense_mtx,0,sizeof(double)*colA*BN);


    float timeForSort = 0;
    ColSort(timeForSort,
            colA,
            nnz,
            BN,
            nnzpercol,
            csccolptr,
            cscrowidx,
            cscval, 

            sortrowidx_tmp,
            sortval_tmp,
            sortnnz_tmp,
             
            dense_matrix,
            sort_dense_mtx);

    int *sortnnz_tmp1= (int *)malloc(sizeof(int)*(colA+1));
    memset(sortnnz_tmp1,0,sizeof(int)*(colA+1));
    for(int i =0; i<colA;i++)
    {
        sortnnz_tmp1[i] = sortnnz_tmp[i];
    }
    exclusive_scan(sortnnz_tmp1, colA+1);

    memset(testcsc_matrix_c,0, sizeof(double) * rowA*BN); 

    for(int i=0; i<colA; i++)
    {
      for(int j=sortnnz_tmp1[i]; j<sortnnz_tmp1[i+1]; j++)
      {
        for(int k=0; k<BN; k++)
        {
          int row_index = sortrowidx_tmp[j];
          int dense_index = i * BN;
          testcsc_matrix_c[row_index *BN +k] += sortval_tmp[j] * sort_dense_mtx[i *BN+k];
        }
      }
    }

    ResultVerify(testcsc_matrix_c, golden_matrix_c,rowA*BN, errorCSC);

#if VERIFYCSC
    int errorsort=0;    
    if(errorCSC !=0)
    {
        printf("error csc sort, error = %d\n", errorsort); 

    }
#endif



float time_irrepart = 0;

    
    double reside_ratio=0;
    formattransation(time_irrepart,
                     matrixA,
                     sortrowidx_tmp,
                     sortval_tmp,
                     sortnnz_tmp,
                     
                     nnz,
                     rowA,
                     colA,
                     reside_ratio);

    double *sort_dense_mtx1 = (double *)malloc(sizeof(double)*colA * BN);  
    memset(sort_dense_mtx1,0,sizeof(double)*colA*BN);
    dense_mtx2dense_mtx_spmm(colA, BN, sort_dense_mtx, sort_dense_mtx1);
                
    double shuffle_ratio = 0;
    block_catgory(matrixA,shuffle_ratio); 


    float timeFormatTran = 0;
    int h_count=0;
    formatTransform(timeFormatTran,
                    matrixA,
                    sortrowidx_tmp,
                    sortval_tmp,
                    sortnnz_tmp,
                    nnz,
                    rowA,
                    colA,
                    h_count);


    double *result_mtx = (double *)malloc(sizeof(double) * rowA*BN);
    memset(result_mtx,0, sizeof(double) * rowA*BN);
    double time_Swift_cpu=0;
    Swift_cpu(time_Swift_cpu,
                 matrixA,
                 nnz,
                 rowA,
                 colA,
                 BN,

                 sort_dense_mtx,
                 result_mtx);

#if VERIFYCSC
    int errorCPU=0;
    ResultVerify(result_mtx, golden_matrix_c, rowA*BN, errorCPU);
    if(errorCPU !=0)
    {
        printf("error Swift CPU, error = %d\n", errorCPU); 
    }
#endif




    float time_Swift_gpu=0;
    double gflops_Swift_gpu=0;

    float time_Swift_gpu1=0;
    double gflops_Swift_gpu1=0;

    float time_Swift_gpu2=0;
    double gflops_Swift_gpu2=0;
    
    float time_Swift_gpu3=0;
    double gflops_Swift_gpu3=0;
    memset(result_mtx,0, sizeof(double) * rowA*BN);     
 



    Swift_gpu1(filename,
               time_Swift_gpu,
               gflops_Swift_gpu,
               matrixA,
               rowA,
               colA,
               BN,
               nnz,
               sort_dense_mtx1,
               result_mtx,
               golden_matrix_c);

    memset(result_mtx,0, sizeof(double) * rowA*BN);




    int reside_n = matrixA->reside_col;
    int reside_nnz = matrixA->reside_nnz;
    int *reside_ptr = matrixA->reside_cscptr;
    int *reside_rowidx = matrixA->reside_cscrowidx;

float time_balance = 0;
timeval tbalance1, tbalance2;
gettimeofday(&tbalance1, NULL);

    balance(matrixA,
             rowA, reside_n, reside_nnz,
             reside_ptr,
             reside_rowidx);

gettimeofday(&tbalance2, NULL); 
time_balance = (tbalance2.tv_sec - tbalance1.tv_sec) * 1000.0 + (tbalance2.tv_usec - tbalance1.tv_usec) / 1000.0;


    Swift_gpu2(filename,
                  time_Swift_gpu1,
                  gflops_Swift_gpu1,
                  matrixA,
                  rowA,
                  colA,
                  BN,
                  nnz,
                  sort_dense_mtx,
                  result_mtx,
                  golden_matrix_c);

    memset(result_mtx,0, sizeof(double) * rowA*BN);
  

    Swift_GPU(filename,
              time_Swift_gpu2,
              gflops_Swift_gpu2,
              matrixA,
              rowA,
              colA,
              BN,
              nnz,
              sort_dense_mtx1,
              result_mtx,
              golden_matrix_c);


    Swift_GPU_3(filename,
              time_Swift_gpu3,
              gflops_Swift_gpu3,
              matrixA,
              rowA,
              colA,
              BN,
              nnz,
              sort_dense_mtx1,
              result_mtx,
              golden_matrix_c);

    int nnz_regu = matrixA->regular_nnz;
    int *tmprowidx1 = (int *)malloc(nnz_regu * sizeof(int));
    memset(tmprowidx1, 0, nnz_regu * sizeof(int));
    int *sortrowidx = matrixA->sortrowidx;
    for(int i=0; i<nnz_regu; i++)
    {
        tmprowidx1[i] = sortrowidx[i];
    }
float time_transform2_1 = 0; 
float time_transform2_2 = 0;
float time_transform2 = 0;                      

    row_idx_sort(matrixA,sort_dense_mtx1,rowA,colA,BN,time_transform2_1,time_transform2_2);         
    time_transform2 = time_transform2_1 + time_transform2_2;


    float time_Swift_gpu4=0;
    double gflops_Swift_gpu4=0;    
    Swift_GPU_4(filename,
              time_Swift_gpu4,
              gflops_Swift_gpu4,
              matrixA,
              rowA,
              colA,
              BN,
              nnz,
              sort_dense_mtx1,
              result_mtx,
              golden_matrix_c);


    float time_final;
    result_check(time_Swift_gpu, time_Swift_gpu1, time_Swift_gpu2, time_final);

    float time_final1;
    result_check(time_final, time_Swift_gpu3, time_Swift_gpu4, time_final1);                  



float time_sum_trans = timeFormatTran + time_irrepart + time_transform2  +time_balance;


    float preprocess_sum = timeForSort +time_sum_trans;
    float ratio1 = timeForSort / preprocess_sum;
    float ratio2 = time_sum_trans / preprocess_sum;
  
    if (time_final1 != -1)
    {
        printf("Sort %f ms, block %f ms\n", timeForSort, time_sum_trans);
        printf("Preprocess %f ms, Process %f ms\n", preprocess_sum, time_final1);
        float sumtime1 = preprocess_sum+ time_final1;
        FILE *fout = fopen(RESULT_FILE_PATH, "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s m %d n %d nnz %d sort %f ratio %f block %f ratio %f preprocess %f final %f overall_time %f \n",
                    filename,rowA, colA, nnz,timeForSort, ratio1,time_sum_trans, ratio2  , preprocess_sum , time_final1,sumtime1);
        fclose(fout);
        
    }
    else
    {
        printf("Swift GPU SpMM Check NO PASS!\n");
        FILE *fout = fopen(RESULT_FILE_PATH, "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "erro Swift %s \n",
                       filename );
        fclose(fout);
    }




    free(dense_matrix);
    free(golden_matrix_c);
    free(cscrowidx);
    free(csccolptr);
    free(nnzpercol);
    free(cscval);
    free(testcsc_matrix_c);
    free(sortrowidx_tmp);
    free(sortval_tmp);
    free(sortnnz_tmp);
    //free(result_mtx);
    free(sort_dense_mtx); 
}