# hyper parameters
batch_size=1024
output_dir="/home/cw862/dsi/MSMARCO_baselines/dsi-cont/"
base_data_dir="/home/cw862/MSMARCO/"
base_data_dir_new='/home/cw862/MSMARCO/new_docs'
for filter_num in 10 100 1000 10000
do
    for epoch in 1 5 10
    do
        model_path="/home/cw862/dsi/MSMARCO_baselines/dsi-cont/dsi_cont_'"$filter_num"'_'"$epoch"'"
        output_name="newdocs_'"$filter_num"'_epoch'"$epoch"'"
        train_cmd="
        python  ../../dsi_model_v1.py \
        --batch_size=$batch_size  --output_dir=$output_dir --output_name=$output_name \
        --model_name='bert-base-uncased' --initialize_model $model_path \
        --base_data_dir=$base_data_dir --base_data_dir_new=$base_data_dir_new --test_only --filter_num=$filter_num"
        echo $train_cmd
        eval $train_cmd
    done
done