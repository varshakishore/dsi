embeddings_path=""
model_path="/home/cw862/DPR/MSMARCO_final_output_2/dpr_biencoder.1"
eval_step=1000
write_path="/home/cw862/DSI/dsi/outputs"
data_dir="/home/vk352/dsi/data/NQ320k"

eval_cmd="
python stepwise_evaluation.py --initialize_embeddings $embeddings_path \
--initialize_model $model_path --eval_step $eval_step \
--dataset 'nq320k' --data_path $data_dir --doc_type 'old_docs'"

echo $eval_cmd
eval $eval_cmd

cp $0 $write_path