python direct_optimize.py --trials 30 --dataset nq320k --optimizer lbfgs --squared_hinge --harmonic_beta 5 --bayesian_target harmonic_mean --multiple_queries --tune_parameters --lbfgs_iterations 30 --train_q --num_qs 15 --write_path_dir "nq320k_results/final_mean_new_q" --mean_new_q