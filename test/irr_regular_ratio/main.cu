#include "coo2csc.h"
#include "sys/time.h"
#include <vector>

#include <iostream>

#define BN 32

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
    strcpy(filename, fileName.c_str());

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
    double *cscval = (double *)malloc(sizeof(double)* (nnz));
    memset(cscval, 0, sizeof(double)*(nnz));
    int *nnzpercol=(int *)malloc(sizeof(int)*(colA));
    memset(nnzpercol, 0 ,sizeof(int)*(colA));

  
    struct timeval transpose1,transpose2;
    double transposetime;
    double csr2cscTime=0;
    gettimeofday(&transpose1,NULL);

    csr2csc(nnz,rowA,colA,csrcolidx,csrrowptr,csrval,&cscrowidx,&csccolptr,&nnzpercol,&cscval);


    gettimeofday(&transpose2,NULL);
    transposetime = (transpose2.tv_sec - transpose1.tv_sec) * 1000.0 + (transpose2.tv_usec - transpose1.tv_usec) / 1000.0;

 

    slide_matrix *matrixA = (slide_matrix *)malloc(sizeof(slide_matrix));

    int *sortrowidx_tmp = (int *)malloc(sizeof(int)*nnz);
    memset(sortrowidx_tmp,0,sizeof(int)*nnz);
    double *sortval_tmp = (double *)malloc(sizeof(double)*nnz);
    memset(sortval_tmp,0,sizeof(double)*nnz);
    int *sortnnz_tmp= (int *)malloc(sizeof(int)*(colA));
    memset(sortnnz_tmp,0,sizeof(int)*colA);
    double *sort_dense_mtx = (double *)malloc(sizeof(double)*colA * BN);  
    memset(sort_dense_mtx,0,sizeof(double)*colA*BN);

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



    

    struct timeval formattransation1,formattransation2;
    double formattransationtime;
    gettimeofday(&formattransation1,NULL);

    formattransation(matrixA,
                     sortrowidx_tmp,
                     sortval_tmp,
                     sortnnz_tmp,
                     
                     nnz,
                     rowA,
                     colA);

    gettimeofday(&formattransation2,NULL);
    formattransationtime = (formattransation2.tv_sec - formattransation1.tv_sec) * 1000.0 + (formattransation2.tv_usec - formattransation1.tv_usec) / 1000.0;

//------------------------------------------------------------------------------------------------------------------------------------------------------     

    free(nnzpercol);
    free(sortrowidx_tmp);
    free(sortval_tmp);
    free(sortnnz_tmp);



    struct timeval classification1,classification2;
    double classificationtime;
    gettimeofday(&classification1,NULL);  

    int regularpart=0;
    int irregularpart=0;
    tile_classification(filename,
                        matrixA,
                        nnz,rowA,colA,
                        regularpart, irregularpart);
                          
    gettimeofday(&classification2,NULL);
    classificationtime = (classification2.tv_sec - classification1.tv_sec) * 1000.0 + (classification2.tv_usec - classification1.tv_usec) / 1000.0;

    int sum = regularpart + irregularpart;
    double ratio1 = (double)regularpart / (double)sum;
    double ratio2 = (double)irregularpart / (double)sum;

    printf("success Classification, regular part: %d %f irregular part: %d %f \n",regularpart, ratio1,irregularpart, ratio2);

    FILE *fout = fopen(RESULT_FILE_PATH, "a");
    if (fout == NULL)
    printf("Writing results fails.\n");
    fprintf(fout, "%s nnz %d regulatpart %d %f irregularpart %d %f \n",
            filename, nnz ,regularpart, ratio1,irregularpart, ratio2);
    fclose(fout);  


}


