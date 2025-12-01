void block_catgory(slide_matrix *matrix, double &shuffle_ratio)
{
    int tilenum = matrix->tilenum;
    matrix->tile_format = (int *)malloc(matrix->tilenum * sizeof(int));
    memset(matrix->tile_format, 0, matrix->tilenum * sizeof(int));
    int *tile_formatc = matrix->tile_format;

    int *sortrowidx = matrix->sortrowidx;
    int *tile_ptr = matrix->tile_ptr;

    for(int i = 0; i< tilenum;i++)
    {
        int flag = 0;
        for(int j = tile_ptr[i]; j<tile_ptr[i+1];j++)
        {

            int rowidx = sortrowidx[tile_ptr[i]];
            if(sortrowidx[j] != rowidx)
            {
                flag = 1;
                break;
            }
  
        }
        if(flag ==0)
        {
            tile_formatc[i] = 0;            
        }
        else
        {
            tile_formatc[i] = 1;
        }
    }
 

    int countt=0;
    for(int i = 0; i< tilenum;i++)
    {
        if(tile_formatc[i] == 0)
        {
            countt++;
        }
    }

    int *count4seg = (int *)malloc((tilenum+1) * sizeof(int));
    memset(count4seg, 0, (tilenum+1) * sizeof(int));    
    for(int i = 0; i< tilenum;i++)
    {
        if(tile_formatc[i] == 1)
        {
            int compare =sortrowidx[tile_ptr[i]];
            for(int j = tile_ptr[i]+1; j<tile_ptr[i+1];j++)
            {
                if(sortrowidx[j] != compare)
                {
                    count4seg[i]++;
                    compare = sortrowidx[j];                    
                }
            }
        }
    }   

    int countt2=0;
    for(int i = 0; i< tilenum;i++)
    {
        if(count4seg[i] < 16 && count4seg[i] > 0)
        {
            tile_formatc[i] = 2;
            countt2++;
        }

    }



    shuffle_ratio = (double)countt/(double)tilenum;


    int *countrow = (int *)malloc((tilenum+1) * sizeof(int));
    memset(countrow,0,(tilenum +1) * sizeof(int));

    for(int i = 0;i < tilenum;i++)
    {
        countrow[i] = 1;
 
        int start = tile_ptr[i];
        int compare = sortrowidx[start];
        for(int j = tile_ptr[i]+1; j<tile_ptr[i+1]; j++)
        {
            if(sortrowidx[j] != compare)
            {
                compare = sortrowidx[j];
                countrow[i]++;
            }
        }
    }
    exclusive_scan(countrow,(tilenum + 1));

    int segsum = countrow[tilenum];
    matrix->segsum = segsum;

    matrix->segmentoffset = (int *)malloc((segsum+1) * sizeof(int));
    memset(matrix->segmentoffset,0,(segsum +1) * sizeof(int));
    int *segmentoffset = matrix->segmentoffset;

    int *segmentoffset_tmp = (int *)malloc(sizeof(int)*(segsum+1));
    memset(segmentoffset_tmp,0,(segsum +1) * sizeof(int));


    for(int i=0;i<tilenum;i++)
    {
        int start = tile_ptr[i];
        int segpoint = countrow[i];
        int compare = sortrowidx[start];
        for(int j = tile_ptr[i]+1;j<tile_ptr[i+1];j++)
        {
            if(sortrowidx[j] == compare)
            {
                segmentoffset[segpoint]++;
                segmentoffset_tmp[segpoint]++;
            }
            else
            {
                compare = sortrowidx[j];
                segpoint++;
            }

        }
    }
    for(int i = 0; i<segsum;i++)
    {
        //segmentoffset[i] +=1;
        segmentoffset_tmp[i] +=1; 
    }

    exclusive_scan(segmentoffset_tmp,(segsum+1));

    int regularnnz = matrix->regular_nnz;
    if(segmentoffset_tmp[segsum] != regularnnz)
    {
        printf("error segmentoffset: error is : %d \n", segmentoffset_tmp[segsum]);
    }


    matrix->sortrowindex = (int *)malloc((regularnnz) * sizeof(int));
    memset(matrix->sortrowindex,0,(regularnnz) * sizeof(int));
    int *sortrowindex = matrix->sortrowindex;

    for(int i =0; i<tilenum;i++)
    {
        int start = countrow[i];
        int stop = countrow[i+1];
        for(int j = start; j<stop; j++)
        {
            int index = segmentoffset_tmp[j];
            int targetval = segmentoffset[j];
            sortrowindex[index] = targetval;
        }
    }

}