#include "mmio.h"
#include "cmath"
#include "cstdlib"
#include "format.h"

#ifndef TILESIZE
#define TILESIZE 32
#endif

#ifndef TIMING
#define TIMING 1
#endif

#ifndef VERIFY
#define VERIFY 1
#endif

#ifndef VERIFY_2
#define VERIFY_2 1
#endif

#define RESULT_FILE_PATH "./data/results_irrOpt_VS_nonOpt_128.txt"

float min_val(float a, float b, float c) {
    return fminf(a, fminf(b, c));
}

void result_check(float time_Swift_gpu, float time_Swift_gpu1, float time_Swift_gpu2, float &time_final)
{
    if(time_Swift_gpu !=-1 && time_Swift_gpu1 !=-1 && time_Swift_gpu2 !=-1)
    {
        time_final = fminf(time_Swift_gpu, fminf(time_Swift_gpu1, time_Swift_gpu2));
    }
    else if(time_Swift_gpu ==-1 && time_Swift_gpu1 !=-1 && time_Swift_gpu2 !=-1)
    {
        time_final = fminf(time_Swift_gpu1, time_Swift_gpu2);
    }
    else if(time_Swift_gpu !=-1 && time_Swift_gpu1 ==-1 && time_Swift_gpu2 !=-1)
    {
        time_final = fminf(time_Swift_gpu, time_Swift_gpu2);
    }
    else if(time_Swift_gpu !=-1 && time_Swift_gpu1 !=-1 && time_Swift_gpu2 ==-1)
    {
        time_final = fminf(time_Swift_gpu, time_Swift_gpu1);
    }
    else if(time_Swift_gpu ==-1 && time_Swift_gpu1 ==-1 && time_Swift_gpu2 ==-1)
    {
        time_final = -1;
    }
}


void dense_mtx2dense_mtx_spmm(int rows, int cols, double *originmtx, double *targetmtx)
{
  int k=0;
  for(int i=0; i<cols;i++)
  {
    for(int j=0; j<rows;j++)
    {
      targetmtx[k]=originmtx[j*cols + i];
      k++;
    }
  }
}

void auxilary_print_mtx(int row, int col,int width, double *mtx)
{
    for(int i=0; i< row; i++)
    {
        for(int j=0; j<col; j++)
        {
            printf("%f ", mtx[i*width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool cmp(Element a, Element b) {
    return a.sortnnzpercol < b.sortnnzpercol;
}

void exclusive_scan(int *input, int length) 
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}  

void exclusive_reduce(int *input, int length)
{
    if(length ==0 || length ==1)
    return;
    int old_val, new_val;
    old_val=input[0];
    for(int i=1; i<length;i++)
    {
        new_val=input[i];
        input[i]=input[i]-old_val;
        old_val=new_val;
     
    }
}

void generateSparseVector(double sparsity, int size, double *vector) {
    // 设置随机数种子
    srand(1);

    for (int i = 0; i < size; ++i) {
        // 生成1-10的随机数
        double randomValue = (rand() % 10 + 1) * 1.0;

        // 根据稀疏度条件将值置为非零
        if ((rand() % 10000) < sparsity * 10000) {
            vector[i] = randomValue;
        } else {
            vector[i] = 0.0;
        }
    }
}

void Golden_y(int rowA, int colA, int *csrPtr, int *csrColidx, double *csrValue, double *VectorX, double *GoldenY)
{
	for (int i = 0; i < rowA; i++)
	{
		double sum = 0;
		for (int j = csrPtr[i]; j < csrPtr[i+1]; j++)
		{
			sum += csrValue[j] * VectorX[csrColidx[j]];
		}
		GoldenY[i] = sum;
	}    

}

void CSC_SpMV(int rowA, int colA, int *cscPtr, int *cscRowidx, double *cscValue, double *VectorX, double *ResultY)
{
	for (int i = 0; i < colA; i++)
	{
		for (int j = cscPtr[i]; j < cscPtr[i+1]; j++)
		{
			int row_tmp = cscRowidx[j];
            double val_tmp = cscValue[j];
            ResultY[row_tmp] += val_tmp * VectorX[i];
		}
	}  
}


void ResultVerify(double *resultA, double *resultB, int length, int &result)
{
    for (int i = 0; i < length; i++)
    {
        //if (abs(resultB[i] - resultA[i]) > 0.01 * abs(resultA[i]))
        if( fabs((resultB[i] - resultA[i])) > 1e-2 )
        //if( !(resultB[i] > resultA[i] || resultB[i] < resultA[i]))
        {
            result++;
        }
    }   
    //printf("verify :%d\n", result) ; 
}

int mmio_allinone(int *m, int *n, int *nnz, int *isSymmetric, 
                  int **csrrowptr, int **csrcolidx, double **csrval,
                  char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp; 

    int ret_code;
    MM_typecode matcode;  
    FILE *f;              

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)  //FILE *fopen(char *filename, *type);  title中的filename，
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    double *csrVal_tmp    = (double *)malloc(nnz_mtx_report * sizeof(double));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    int *csrRowPtr_alias = (int *)malloc((m_tmp+1) * sizeof(int));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    double *csrVal_alias    = (double *)malloc(nnz_tmp * sizeof(double));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {            
            int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrrowptr = csrRowPtr_alias;
    *csrcolidx = csrColIdx_alias;
    *csrval = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}