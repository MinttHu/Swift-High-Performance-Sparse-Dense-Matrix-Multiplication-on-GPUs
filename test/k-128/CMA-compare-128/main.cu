#include <iostream>
#include <vector>
#include "sys/time.h"
//#include "absl/random/random.h"

#include "coo2csc.h"
#include "Swift_cpu.h"
#include "gpu_csr2csc.h"
#include "gpu_cusparse_spmm.h"
#include "Swift_gpu_row_major.h"
#include "Swift.h"

#include "balance.h"
#include "block_catgory.h"

#include "ColSort_cuda.h"
#include "formatTransform_cuda.h"
#define BN 128

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
/*
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
*/

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


    std::snprintf(filename, 256, "%s", fileName.c_str());

//|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-

    printf("read success,input matrix A :(%i,%i) nnz =%i  width=%d \n",rowA,colA,nnz,BN);
  

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

    double csr2cscTime = 0;
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
    else
    {
        printf("success format csc\n");
    }
#endif


    slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));

    int *sortrowidx_tmp = (int *)malloc(sizeof(int)*nnz);

    double *sortval_tmp = (double *)malloc(sizeof(double)*nnz);

    int *sortnnz_tmp= (int *)malloc(sizeof(int)*(colA+1));

    double *sort_dense_mtx = (double *)malloc(sizeof(double)*colA * BN);  
    memset(sort_dense_mtx,0,sizeof(double)*colA*BN);

double time_colsort = 0;

timeval tcolsort1, tcolsort2;

gettimeofday(&tcolsort1, NULL);

    col_sort(colA,
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


gettimeofday(&tcolsort2, NULL);  
time_colsort = (tcolsort2.tv_sec - tcolsort1.tv_sec) * 1000.0 + (tcolsort2.tv_usec - tcolsort1.tv_usec) / 1000.0;




double time_transform = 0;
double time_irrepart = 0;
timeval ttransform1, ttransform2;
gettimeofday(&ttransform1, NULL);

    formattransation(time_irrepart,
                     matrixA,
                     sortrowidx_tmp,
                     sortval_tmp,
                     sortnnz_tmp,
                     
                     nnz,
                     rowA,
                     colA);


    block_catgory(matrixA);

gettimeofday(&ttransform2, NULL); 
time_transform = (ttransform2.tv_sec - ttransform1.tv_sec) * 1000.0 + (ttransform2.tv_usec - ttransform1.tv_usec) / 1000.0;


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
    else
    {
        printf("success Swift CPU time: %f ms\n",time_Swift_cpu);
    }
#endif




    float time_Swift_gpu=0;
    double gflops_Swift_gpu=0;

    float time_Swift_gpu1=0;
    double gflops_Swift_gpu1=0;

    float time_Swift_gpu2=0;
    double gflops_Swift_gpu2=0;
    memset(result_mtx,0, sizeof(double) * rowA*BN);     

    double *sort_dense_mtx1 = (double *)malloc(sizeof(double)*colA * BN);  
    memset(sort_dense_mtx1,0,sizeof(double)*colA*BN);
    dense_mtx2dense_mtx_spmm(colA, BN, sort_dense_mtx, sort_dense_mtx1);


    int reside_n = matrixA->reside_col;
    int reside_nnz = matrixA->reside_nnz;
    int *reside_ptr = matrixA->reside_cscptr;
    int *reside_rowidx = matrixA->reside_cscrowidx;

double time_balance = 0;
timeval tbalance1, tbalance2;
gettimeofday(&tbalance1, NULL);

    balance(matrixA,
             rowA, reside_n, reside_nnz,
             reside_ptr,
             reside_rowidx);

gettimeofday(&tbalance2, NULL); 
time_balance = (tbalance2.tv_sec - tbalance1.tv_sec) * 1000.0 + (tbalance2.tv_usec - tbalance1.tv_usec) / 1000.0;



    Swift_row_major(filename,
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

    FILE *fout = fopen(RESULT_FILE_PATH, "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s m %d n %d width %d nnz %d withoutCMA %f withCMA %f \n",
            filename,rowA, colA, BN, nnz,time_Swift_gpu1, time_Swift_gpu2);
    fclose(fout);



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