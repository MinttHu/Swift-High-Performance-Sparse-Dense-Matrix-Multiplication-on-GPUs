void balance(slide_matrix *matrix,
             int m, int n, int nnz,
             int *cscptr,
             int *cscrowidx)
{
    
    
    int numblk_tmp = 0;

    for(int i = 0 ; i < n ; i++)
    {
        int nnzpercol = cscptr[i+1] - cscptr[i];
        if(nnzpercol <= TILESIZE)
        {
            numblk_tmp++;
        }
        else
        {
            numblk_tmp += ceil((double)nnzpercol / (double)TILESIZE);

        }
    }    
    //printf("%d\n",numblk_tmp);
    matrix->numblk = numblk_tmp; 
    matrix->colorblk= (unsigned int *)malloc(sizeof(unsigned int) * numblk_tmp);
    memset(matrix->colorblk , 0 , sizeof(unsigned int) * numblk_tmp); 
    matrix->blkstart = (int *)malloc(sizeof(int) * numblk_tmp);
    memset(matrix->blkstart, 0 , sizeof(int) * numblk_tmp);
    matrix->blkstop = (int *)malloc(sizeof(int) * numblk_tmp);
    memset(matrix->blkstop, 0 , sizeof(int) * numblk_tmp);
    matrix->blkptr = (int *)malloc(sizeof(int) * numblk_tmp);
    memset(matrix->blkptr , 0 , sizeof(int) * numblk_tmp);

    unsigned int *colofblk_tmp = matrix->colorblk;
    int *blkstart_tmp = matrix->blkstart;
    int *blkstop_tmp = matrix->blkstop;
    int *blkptr_tmp = matrix->blkptr;

    int count4blk = 0;
    for(int i = 0 ; i < n ; i++)
    {
        int nnzpercol = cscptr[i+1] - cscptr[i];
        if(nnzpercol <= TILESIZE)
        {
            colofblk_tmp[count4blk] = i;
            blkptr_tmp[i] = count4blk;
            count4blk++;
        }
        else
        {
            int numofblk = ceil((double)nnzpercol / (double)TILESIZE);
            int lenblk = ceil((double)nnzpercol / (double)numofblk);
            blkptr_tmp[i] =  count4blk;
            for(int j = 0 ; j < numofblk ; j++)
            {
                colofblk_tmp[count4blk] = i | 0x80000000;
                blkstart_tmp[count4blk] = cscptr[i] + j * lenblk;
                if(j == numofblk - 1)
                {
                    blkstop_tmp[count4blk] = cscptr[i] + nnzpercol;
                }
                else
                {
                    blkstop_tmp[count4blk] = cscptr[i] + (j+1) * lenblk;
                }
                count4blk++;
            } 
        }
    }
    //printf("count4blk:%d\n",count4blk);

}