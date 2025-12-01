#include <iostream>
#include <vector>
#include "sys/time.h"
#include "auxiliary.h"

#include "row_major_gpu.h"
#include "col_major_gpu.h"

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
    strcpy(filename, fileName.c_str());
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

    colA = 0;
    colA = ceil(( double)nnz / (double)rowA); 
    double squareRootDouble = std::sqrt(nnz);
    rowA = static_cast<int>(squareRootDouble);
    colA = rowA;

    printf("read success,input matrix %s :(%i,%i) nnz =%i  width=%d \n", filename ,rowA,colA,rowA*colA,BN);
  
    double *matrixA = (double *)malloc(sizeof(double) * rowA * colA);
    memset(matrixA,0, sizeof(double ) * rowA *colA);

    double *dense_matrix = (double *)malloc(sizeof(double) * colA*BN);
    memset(dense_matrix,0, sizeof(double) * colA*BN);

    unsigned seed;
    seed = time(0);
    srand(seed);
    for (int i=0; i<colA*BN; i++) 
    {
        dense_matrix[i] = double(rand() %100 - 50)/100;
    }

    for (int i=0; i<rowA * colA; i++) 
    {
        matrixA[i] = double(rand() %1000 - 500)/1000;
        //csrval[i] = i % 10 + 1;
    }

    double *dense_matrix1 = (double *)malloc(sizeof(double) * colA*BN);
    memset(dense_matrix1,0, sizeof(double) * colA*BN);
    dense_mtx2dense_mtx_spmm(colA, BN, dense_matrix, dense_matrix1);



    double *result_mtx = (double *)malloc(sizeof(double) * rowA*BN);
    memset(result_mtx,0, sizeof(double) * rowA*BN);



    float time_Swift_gpu=0;
    double gflops_Swift_gpu=0;

    float time_Swift_gpu1=0;
    double gflops_Swift_gpu1=0;

    Swift_gpu(filename,
              time_Swift_gpu,
              gflops_Swift_gpu,
              rowA,
              colA,
              BN,
              nnz,
              matrixA,
              dense_matrix,
              result_mtx);

    memset(result_mtx,0, sizeof(double) * rowA*BN);


    Swift_gpu1(filename,
                 time_Swift_gpu1,
                 gflops_Swift_gpu1,
                 rowA,
                 colA,
                 BN,
                 nnz,
                 matrixA,
                 dense_matrix1,
                 result_mtx);



    if (time_Swift_gpu != -1 && time_Swift_gpu1 != -1)
    {
        printf("Ideal without CMA SpMM GPU time = %f ms\n",time_Swift_gpu);
        printf("Ideal with CMA SpMM GPU time = %f ms\n",time_Swift_gpu1);
     
        FILE *fout = fopen(RESULT_FILE_PATH, "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "%s m %d n %d width %d nnz %d Ideal_without_CMA %f Ideal_with_CMA %f \n",
            filename,rowA, colA, BN, rowA*colA,time_Swift_gpu, time_Swift_gpu1);
        fclose(fout);
        
    }
    else
    {
        printf(" GPU Check NO PASS!\n");
        FILE *fout = fopen(RESULT_FILE_PATH, "a");
        if (fout == NULL)
            printf("Writing results fails.\n");
        fprintf(fout, "erro Swift (dense mtx col-major)%s \n",
                       filename );
        fclose(fout);
    }

    free(dense_matrix);
    free(dense_matrix1);


    free(result_mtx);
}