# hyper parameters

batch_size=128
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.001
train_epochs=20
output_dir="/home/cw862/DSI/dsi/outputs/MSMARCO"
logging_step=200
model_path="/home/cw862/DPR/MSMARCO_final_output/dpr_biencoder.0"
base_data_dir="/home/cw862/MSMARCO/old_docs"

train_cmd="
python  dsi_model_v1.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
--freeze_base_model --initialize_model $model_path --base_data_dir $base_data_dir"

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory"
cp $0 $output_dir