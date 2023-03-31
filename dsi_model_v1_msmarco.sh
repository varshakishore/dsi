# hyper parameters
batch_size=1024
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.0005
train_epochs=20
output_dir="/home/cw862/DSI/dsi/MSMARCO_baselines/dpr/"
logging_step=200
data_dir="/home/cw862/MSMARCO"
base_data_dir_new='/home/cw862/MSMARCO/new_docs'
model_path="/home/cw862/DPR/MSMARCO_final_output_2/dpr_biencoder.1"
wandb_name="freeze_100"


train_cmd="
python  dsi_model_v1.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate=$learning_rate  --train_epochs=$train_epochs \
--initialize_model=$model_path --model_name='bert-base-uncased' \
--wandb_name=$wandb_name --output_name=$wandb_name --base_data_dir_new=$base_data_dir_new \
--base_data_dir=$data_dir --filter_num=100 --freeze_base_model"

# --freeze_base_model --initialize_model $model_path 

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory to:"
echo $output_dir
cp $0 $output_dir