CMA-compare:
  1 ./Data_process.sh
  2 ./speedup.py
  2 ./speedup_129.py
  3 ./cma_compare.py
  4 ./cma_compare_128.py

FP32:
  Time comparasion of Swift with SOTA
  1 ./Data_process.sh
  2 ./scatterplot.py


  Geometric mean Speedup of Swift with SOTA
  1 ./gemean.sh
  2 ./gm_ave.py

  Speedup Distribution of Swift over SOTA
  1 ./speedup.py
  2 ./speedup_scatter.py

FP64:
  Same as FP32

idealCMA
  Same as CMA-compare

irr_opt
  ./hisgram.py

irr_regular_ratio
  ./draw

irregular_K
  1 ./Data_process.sh
  2 ./gm_ave.py
  3 replace the data on polyline.py (Line 6 and Line 7)
  4 ./polyline.py

LoadBalance
  1 Replace the data on polyline.py (Line 7 and Line 8)
  2 ./polyline.py

MTX_info
  ./Data_process.sh
    (Performance relation of ASpT-Swift with distribution)
  1 ./ASpT_Swift.sh
  2 ./ASpT_Swift_scatter.py

    (Relation of average speedup of Swift over SOTA with distribution)
  1 ./blkratio.sh
  2 ./blk_scatter.py
    (Relation of speedup with sparsity)
  1 ./sparsity.sh
  2 ./gm_ave.py
  3 Replace the data on polyline.py
  4 ./polyline.py

preprocess
  1 ./Data_process.sh
  2 ./block_sort.py
  3 ./time_compare.py

  