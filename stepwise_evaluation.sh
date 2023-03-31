for emb_path in "/home/jl3353/dsi/nq320k_results/final_seed45/2023-01-26_10-46-12" #"/home/jl3353/dsi/nq320k_results/final_seed46/2023-01-26_10-47-12" #"/home/jl3353/dsi/nq320k_results/final_seed43/2023-01-26_10-02-59" "/home/jl3353/dsi/nq320k_results/final_seed44/2023-01-26_10-19-42" "/home/jl3353/dsi/nq320k_results/final_seed47/2023-01-26_10-03-24" "/home/jl3353/dsi/nq320k_results/final_seed48/2023-01-26_10-20-08" "/home/jl3353/dsi/nq320k_results/final_seed49/2023-01-26_10-04-16" "/home/jl3353/dsi/nq320k_results/final_seed50/2023-01-26_10-21-12"
do
    embeddings_path=$emb_path
    model_path="/home/vk352/dsi/NQ320k_outputs/old_docs/finetune_old_epoch17"
    eval_step=1000
    write_path=$embeddings_path
    data_dir="/home/vk352/dsi/data/NQ320k"

    # model_path="/home/cw862/DSI/dsi/outputs/MSMARCO_2_bs1600lr5e-4/finetune_old_epoch1"
    # data_dir="/home/cw862/MSMARCO/"

    eval_cmd="
    python stepwise_evaluation.py --initialize_embeddings $embeddings_path \
    --initialize_model $model_path --eval_step $eval_step \
    --dataset 'NQ320k' --base_data_dir $data_dir --doc_type 'old_docs' --write_path $write_path --frequent"

    echo $eval_cmd
    eval $eval_cmd
done
