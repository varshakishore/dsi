# hyper parameters

batch_size=1024
# set it to 0.0001 for bert and 0.001 for t5
output_dir="/home/vk352/dsi/NQ320k_outputs/temp/"
logging_step=200
model_path="/home/vk352/dsi/msmarco_baselines/dpr_0.0005/finetune_new_10_epoch10"
base_data_dir="/home/cw862/MSMARCO/"
base_data_dir_new='/home/cw862/MSMARCO/new_docs'

train_cmd="
python  dsi_model_v1.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--model_name='bert-base-uncased' \
--initialize_model $model_path --base_data_dir_new=$base_data_dir_new --test_only --filter_num=10 \
--base_data_dir=$base_data_dir"

echo $train_cmdls
eval $train_cmd

# echo "copy current script to model directory to:"
# echo $output_dir
# cp $0 $output_dir