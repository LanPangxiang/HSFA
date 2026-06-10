
for data in NYC CA TKY; do
 for k in 2 3 4 5 6 7 8 9 10; do
   for ((c=1; c<=k-1; c++)); do
     echo "Running data=$data, K=$k, c=$c"
     python main.py \
       --data_name "$data" \
       --pre_hsfa_num_bands "$k" \
       --pre_hsfa_low_band_count "$c" --batch 128 --epoch 50 --cuda True --gpu_id 0
   done
 done
done


