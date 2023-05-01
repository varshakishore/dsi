# hyper parameters
batch_size=128
# default lr from the tfew implementation
learning_rate=0.003
train_epochs=50
# output_dir="/home/vk352/dsi/string_dsi/my_imp/"
output_dir="/home/cw862/dsi/string_dsi/semantic_id_lora/"
logging_step=200
model_path="/home/vk352/dsi/string_dsi/semantic_id/finetune_old_epoch45"
base_data_dir_new='/home/vk352/dsi/data/NQ320k/new_docs'
output_name='finetune_new_lora_50_epoch'
wandb_name="semantic_id_new_50_lora"
semantic_id_path="/home/vk352/dsi/data/semantic_id_map_30"

train_cmd="
python dsi_model_string_v1.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='t5-base' \
--semantic_id_path=$semantic_id_path --new_only --output_name=$output_name \
--base_data_dir_new=$base_data_dir_new --initialize_model $model_path \
--wandb_name $wandb_name --lora "
# --wandb_name $wandb_name 
# --freeze_base_model --initialize_model $model_path --semantic_id_path=$semantic_id_path --averaging_path=$averaging_path
# --ewc
echo $train_cmd
eval $train_cmd

echo "copy current script to model directory to:"
echo $output_dir
cp $0 $output_dir