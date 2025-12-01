#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <omp.h>





void row_idx_sort1(slide_matrix *matrix, double *dense_mtx, int rowA, int colA, int width)
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

    int *back_rowidx = (int *)malloc(nnz_regu * sizeof(int));
    memset(back_rowidx, 0, nnz_regu * sizeof(int));

    int loop = tilenum / 8;

    // ==========================
    // 并行排序部分
    // ==========================
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loop; i++)
    {
        int start = i * 8;
        int stop = start + 8;
        int count = 0;
        int temprowidx[256];
        IndexedValue temp[256];

        for (int j = tile_ptr[start]; j < tile_ptr[stop]; j++)
        {
            temprowidx[count] = sortrowidx[j];
            count++;
        }

        if (count != 256)
        {
            #pragma omp critical
            printf("error load rowidx to temprowidx: count: %d\n", count);
        }

        for (int k = 0; k < 256; ++k)
        {
            temp[k].value = temprowidx[k];
            temp[k].original_index = k;
        }

        std::sort(temp, temp + 256, compare);

        for (int k = 0; k < 256; ++k)
        {
            int idx = tile_ptr[start] + k;
            back_rowidx[idx] = temp[k].value;
            val_position[tile_ptr[start] + temp[k].original_index] = k;
        }
    }

    // ==========================
    // 余数部分（不并行）
    // ==========================
    if (tilenum % 8 != 0)
    {
        int start = (tilenum / 8) * 8;
        int stop = tilenum;
        int numEle = (stop - start) * 32;
        IndexedValue *temp1 = (IndexedValue *)malloc(numEle * sizeof(IndexedValue));
        int *temprowidx1 = (int *)malloc(numEle * sizeof(int));
        memset(temprowidx1, 0, numEle * sizeof(int));

        int count = 0;
        for (int j = tile_ptr[start]; j < tile_ptr[stop]; j++)
        {
            temprowidx1[count] = sortrowidx[j];
            count++;
        }

        if (count != numEle)
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

        free(temp1);
        free(temprowidx1);
    }

    // ==========================
    // 更新 sortrowidx
    // ==========================
    for (int i = 0; i < nnz_regu; i++)
        sortrowidx[i] = back_rowidx[i];

    // ==========================
    // val_offset 计算（串行）
    // ==========================
    for (int i = 0; i < loop; i++)
    {
        int start = i * 8;
        int stop = start + 8;
        int count = 0;
        int compared_val = sortrowidx[tile_ptr[start]];
        for (int j = tile_ptr[start] + 1; j < tile_ptr[stop]; j++)
        {
            int val2 = sortrowidx[j];
            if (compared_val == val2)
            {
                count++;
                val_offset[j - count]++;
            }
            else
            {
                if (count == 0)
                    val_offset[j - 1] = -1;
                compared_val = val2;
                count = 0;
            }
        }
    }

    free(back_rowidx);
}
