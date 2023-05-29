# hyper parameters
batch_size=1024
learning_rate=0.00005
train_epochs=20
output_dir="/home/cw862/dsi/MSMARCO_baselines/dsi-cont/"
logging_step=200
data_dir="/home/cw862/MSMARCO"
base_data_dir_new='/home/cw862/MSMARCO/new_docs'
model_path="/home/cw862/dsi/MSMARCO_baselines/scratch_models/olddocsonly/olddocsonly_0.00005_8"
new_num=10000
wandb_name="dsi_cont_'"$new_num"'_"


train_cmd="
python  ../../dsi_model_v1.py \
--base_data_dir=$data_dir --output_dir=$output_dir --base_data_dir_new=$base_data_dir_new \
--logging_step=$logging_step --learning_rate=$learning_rate  --train_epochs=$train_epochs --batch_size=$batch_size \
--initialize_model=$model_path --model_name='bert-base-uncased' \
--wandb_name=$wandb_name --output_name=$wandb_name  \
--filter_num=$new_num"

echo $train_cmd
eval $train_cmd