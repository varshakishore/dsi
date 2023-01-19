# hyper parameters

output_dir="/home/jl3353/dsi/NQ320k_outputs/old_docs/finetune_old_epoch17"

model_path="/home/vk352/dsi/NQ320k_outputs/old_docs/finetune_old_epoch17"

for split in 'train' 'val' 'test'
do
    train_cmd="
    python save_embeddings.py --output_dir=$output_dir \
    --model_name='bert-base-uncased' --split=$split \
    --initialize_model=$model_path --dataset 'nq320k'"

    echo $train_cmd
    eval $train_cmd

    train_cmd="
    python save_embeddings.py --output_dir=$output_dir \
    --model_name='bert-base-uncased' --split=$split \
    --initialize_model=$model_path --dataset 'nq320k' --generated"

    echo $train_cmd
    eval $train_cmd
done

# train_cmd="
# python  save_embeddings.py --output_dir=$output_dir \
# --model_name='bert-base-uncased' --split='val' \
# --initialize_model=$model_path "

# echo "copy current script to model directory"
# cp $0 $output_dir