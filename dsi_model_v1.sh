# hyper parameters

batch_size=128
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.001
train_epochs=2
output_dir="/home/vk352/dsi/NQ320k_outputs/old_docs_trt2/"
logging_step=200
# model_path="/home/vk352/ANCE/data/NQ320k_v2_dpr2/ann_NQ_test/checkpoint-15000"
model_path="/home/cw862/DPR/NQ320k_final_output/dpr_biencoder.1"
wandb_name="old_docs_trt2"

train_cmd="
python  dsi_model_v1.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
--freeze_base_model --initialize_model $model_path  --wandb_name $wandb_name"

# --freeze_base_model --initialize_model $model_path 

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory to:"
echo $output_dir
cp $0 $output_dir