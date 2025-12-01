#include <cuda_runtime_api.h>
#include <cusparse.h>

void cusparse_spmm(float &time_cusparse_pre,float &time_cusparse, 
                   int A_num_rows, int A_num_cols, int A_nnz,
                   int B_num_rows, int B_num_cols, 
                   int *hA_csrOffsets, int *hA_columns, double *hA_values,
                   double *hB,
                   double *hC)
{
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
/*
    float hC_result[]     = { 19.0f,  8.0f,  51.0f,  52.0f,
                              43.0f, 24.0f, 123.0f, 120.0f,
                              67.0f, 40.0f, 195.0f, 188.0f };
*/
    double alpha           = 1.0;
    double beta            = 0.0;

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    double *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(double))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(double)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
cudaEvent_t event1,event2;
cudaEventCreate(&event1);
cudaEventCreate(&event2);

cudaDeviceSynchronize();

cudaEventRecord(event1,0);
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
cudaEventRecord(event2,0);

cudaEventSynchronize(event1);
cudaEventSynchronize(event2);
cudaEventElapsedTime(&time_cusparse_pre, event1, event2);

cudaDeviceSynchronize();

//cudaEventRecord(event1,0);
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_64F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_64F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                 CUSPARSE_SPMM_CSR_ALG1, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
cudaEventRecord(event1,0);
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                 CUSPARSE_SPMM_CSR_ALG1, dBuffer) )
cudaEventRecord(event2,0);


cudaEventSynchronize(event1);
cudaEventSynchronize(event2);
cudaEventElapsedTime(&time_cusparse, event1, event2);

cudaDeviceSynchronize();

//printf("processtime: %f\n", time_cusparse);
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(double),
                           cudaMemcpyDeviceToHost) )
    /*
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    if (correct)
        printf("spmm_csr_example test PASSED\n");
    else
        printf("spmm_csr_example test FAILED: wrong result\n");
    */
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    //return EXIT_SUCCESS;
}