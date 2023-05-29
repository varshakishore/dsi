# hyper parameters
batch_size=1024
learning_rate=0.00005
train_epochs=10
output_dir="/home/cw862/dsi/MSMARCO_baselines/scratch_models/olddocsonly/"
logging_step=200
base_data_dir="/home/cw862/MSMARCO/"
wandb_name="olddocsonly_'"$learning_rate"'_"

train_cmd="
python  ../../dsi_model_v1.py \
--base_data_dir $base_data_dir --output_dir=$output_dir --wandb_name $wandb_name \
--output_name $wandb_name \
--model_name='bert-base-uncased' \
--batch_size=$batch_size  --learning_rate $learning_rate \
--train_epochs $train_epochs --logging_step=$logging_step"

echo $train_cmd
eval $train_cmd

# echo "copy current script to model directory to:"
# echo $output_dir
# cp $0 $output_dir