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



void row_idx_sort(slide_matrix *matrix, double *dense_mtx,int rowA, int colA, int width)
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

    int *back_rowidx2 = (int *)malloc(nnz_regu * sizeof(int));
    memset(back_rowidx2, 0, nnz_regu * sizeof(int));


    IndexedValue temp[256];

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
      

    int num0=0,num1=0,num2=0;    
    for(int i = 0; i < nnz_regu; i++)
    {
        int val = val_offset[i];
        if(val == -1)
        {
            num0++;
        }
        else if(val >= 1 && val <= 10)
        {
            num1++;
        }
        else if(val > 10)
        {
            num2++;
        }    
    }

    free(temprowidx);
    free(back_rowidx);

}
