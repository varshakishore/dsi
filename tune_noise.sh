#!/bin/bash
for noise in 0.001 0.01 0.1 0.5 1
do 
    echo "python direct_optimize.py --multiple_queries --num_qs 10 --train_q --add_noise --lr 0.0017538651549919983 --lam 3 --m1 0 --m2 0 --write_path_dir '/home/cw862/DSI/dsi/train_10q_Gnoise_$noise'"
    eval "python direct_optimize.py --multiple_queries --num_qs 10 --train_q --add_noise --lr 0.0017538651549919983 --lam 3 --m1 0 --m2 0 --write_path_dir '/home/cw862/DSI/dsi/train_10q_Gnoise_$noise'"

done

