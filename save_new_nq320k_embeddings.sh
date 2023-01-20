# hyper parameters

output_dir="/home/jl3353/dsi/NQ320k_outputs/finetune_old_epoch17"

model_path="/home/vk352/dsi/NQ320k_outputs/old_docs/finetune_old_epoch17"

for doc_split in 'old' 'new' 'tune'
do
    for split in 'train' 'val' 'test' 'gen'
    do
        train_cmd="
        python save_embeddings.py --output_dir=$output_dir \
        --model_name='bert-base-uncased' --split=$split \
        --initialize_model=$model_path --dataset 'nq320k' --doc_split=$doc_split"

        echo $train_cmd
        eval $train_cmd
    done
done

# train_cmd="
# python  save_embeddings.py --output_dir=$output_dir \
# --model_name='bert-base-uncased' --split='val' \
# --initialize_model=$model_path "

# echo "copy current script to model directory"
# cp $0 $output_dir