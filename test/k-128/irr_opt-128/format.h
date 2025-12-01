
#include <string.h>
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
    int *segptr;
    int *seginfo;
    int *seglength;
    int *sortrowindex;
    int tilenum;
    int segsum;

    int regular_nnz;

    int reside_col;
    int reside_nnz;
    int *reside_cscptr;
    int *reside_cscrowidx;
    double *reside_val;
    
    int numblk;
    unsigned int *colorblk;
    int *blkstart;
    int *blkstop;
    int *blkptr;
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
    free(matrix->segptr);
    free(matrix->seginfo);
    free(matrix->seglength);
    free(matrix->sortrowindex);

    free(matrix->reside_cscptr);
    free(matrix->reside_cscrowidx);
    free(matrix->reside_val);
    free(matrix->colorblk);
    free(matrix->blkstart);
    free(matrix->blkstop);
    free(matrix->blkptr);
}

#ifndef slidesize
#define slidesize 32
#endif

#ifndef warpperblock
#define warpperblock 2
#endif

