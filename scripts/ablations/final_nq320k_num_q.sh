for num_q in 5 10
do
    train_cmd="
    python direct_optimize.py --trials 30 --dataset nq320k --optimizer lbfgs --squared_hinge --harmonic_beta 5 --bayesian_target harmonic_mean --multiple_queries --tune_parameters --lbfgs_iterations 30 --train_q --num_qs $num_q --write_path_dir nq320k_results/final_num_q_$num_q --mean_new_q"

    echo $train_cmd
    eval $train_cmd
done

