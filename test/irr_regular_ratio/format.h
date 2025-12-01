#include <math.h>
#include "omp.h"

typedef struct
{
    double *sortval;
    int *sortrowidx;
    int *tile_ptr;
    int *tile_len;
    int *tile_colidx;
    int *tile_tall;
    int *tile_format;
    int *countrow;
    int *segmentoffset;
    int *sortrowindex;
    int tilenum;
    int segsum;
} slide_matrix;


typedef struct Element {
    int sortnnzpercol;
    int index;
} sortA;


void Tile_destroy(slide_matrix *matrix)
{

    free(matrix->sortval);
    free(matrix->sortrowidx);
    free(matrix->tile_ptr);
    free(matrix->tile_len);
    free(matrix->tile_tall);
    free(matrix->tile_colidx);
    free(matrix->tile_format);
    free(matrix->segmentoffset);
    free(matrix->countrow);
    free(matrix->sortrowindex);
}

#ifndef slidesize
#define slidesize 32
#endif

#ifndef warpperblock
#define warpperblock 2
#endif

