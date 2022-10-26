# hyper parameters

batch_size=1600
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.0001
train_epochs=20
output_dir="/home/vk352/dsi/outputs/temp/"
logging_step=20
# embedding="/home/cw862/DSI/data/nq320k_passagesembedding_avg.pkl"

train_cmd="
python  DSI_projection.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
"

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory"
cp $0 $output_dir