# hyper parameters

batch_size=1600
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.001
train_epochs=20
output_dir="/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed_new/"
logging_step=20
embedding_path="/home/vk352/ANCE/data/NQ320k_v2_dpr2"
# model_path="/home/vk352/ANCE/data/NQ320k_v2_dpr2/ann_NQ_test/checkpoint-15000"
model_path="/home/vk352/dsi/outputs/dpr5_finetune_0.001_filtered_fixed/projection_nq320k_epoch15"
optimized_embeddings_path="/home/vk352/dsi/multi_query/classifier_layer.pkl"

train_cmd="
python dsi_model_continual.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
--freeze_base_model --initialize_model=$model_path \
--initialize_embeddings $embedding_path --optimized_embeddings $optimized_embeddings_path"

echo $train_cmd
eval $train_cmd

# echo "copy current script to model directory"
# cp $0 $output_dir