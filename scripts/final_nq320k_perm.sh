for seed in 46
do
    python direct_optimize.py --dataset nq320k --optimizer lbfgs --lr 1 --squared_hinge --harmonic_beta 5 --bayesian_target harmonic_mean --multiple_queries --lbfgs_iterations 30 --train_q --num_qs 15 --write_path_dir "nq320k_results/final_seed$seed" --mean_new_q --m1 6.54905543374915 --m2 6.935875308271506 --lam 0.4842382764359751 --l2_reg 3.144051644302285e-09 --permutation_seed $seed
done