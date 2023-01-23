# hyper parameters

batch_size=128
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.0005
train_epochs=10
output_dir="/home/vk352/dsi/NQ320k_outputs/old_docs_trt2/"
logging_step=200
model_path="/home/vk352/dsi/NQ320k_outputs/old_docs/finetune_old_epoch17"
base_data_dir_new='/home/vk352/dsi/data/NQ320k/new_docs'
output_name='finetune_new_epoch'
# output_name='finetune_new_only_gen_new_epoch'
wandb_name="old_docs_trt2_cont"

train_cmd="
python  dsi_model_v1.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
--initialize_model $model_path --base_data_dir_new=$base_data_dir_new --output_name=$output_name --wandb_name $wandb_name"

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory to:"
# echo $output_dir
# cp $0 $output_dir