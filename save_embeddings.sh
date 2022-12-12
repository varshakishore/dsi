# hyper parameters

output_dir="/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed/nq320k_gen_passage_embeddings.pkl"
model_path="/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed/projection_nq320k_epoch15"

train_cmd="
python  save_embeddings.py --output_dir=$output_dir \
--model_name='bert-base-uncased' \
--initialize_model=$model_path "

echo $train_cmd
eval $train_cmd

# echo "copy current script to model directory"
# cp $0 $output_dir