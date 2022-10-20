# hyper parameters

batch_size=1600
learning_rate=0.001
train_epochs=20
output_dir="/home/cw862/DSI/projection_BERT_nq320k_lre-3/"
logging_step=20
# embedding="/home/cw862/DSI/data/nq320k_passagesembedding_avg.pkl"

train_cmd="
python  BERT_projection.py \
--batch_size=$batch_size  --output_dir=$output_dir --logging_step=$logging_step \
--learning_rate $learning_rate  --train_epochs $train_epochs\
"

echo $train_cmd
eval $train_cmd