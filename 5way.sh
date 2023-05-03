# hyper parameters

batch_size=128
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.0005
train_epochs=1
output_dir="/home/jcl354/dsi/NQ320k_baselines/5way_no_freeze/"
logging_step=200
model_path="/home/vk352/dsi/NQ320k_baselines/scratch_0.00001/finetune_old_epoch20"
wandb_name="5way"


train_cmd="
python  5wayclassification.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='T5-base'     \
--initialize_model $model_path --wandb_name $wandb_name"
#  --freeze_base_model 
#  

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory to:"
echo $output_dir
cp $0 $output_dir