# hyper parameters

batch_size=1600
# set it to 0.0001 for bert and 0.001 for t5
learning_rate=0.001
train_epochs=20
output_dir="/home/vk352/dsi/outputs/dpr5_olddocs_finetune_0.001_filtered_fixed/"
logging_step=20
# embedding="/home/cw862/DSI/data/nq320k_passagesembedding_avg.pkl"
# model_path="/home/vk352/ANCE/data/NQ320k_v2_dpr2/ann_NQ_test/checkpoint-15000"
model_path="/home/cw862/DPR/NQ320k_v2_100k_2/dpr_biencoder.5"

train_cmd="
python  dsi_model.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs --model_name='bert-base-uncased' \
--freeze_base_model --initialize_model $model_path --old_docs_only"

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory"
cp $0 $output_dir