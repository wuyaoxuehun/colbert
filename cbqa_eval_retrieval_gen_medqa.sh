#!/bin/bash

command=$1
gpu='6,7'
#gpu='2,3,4'
#gpu='0,1,2'
#gpu='2,3,4'
gpu='0,1,2,4'
#index_gpu="1,2,3,4"
server_gpu="1"
#gpu='2'
IFS=','
read -ra arr <<< "$gpu"
gpunum=${#arr[@]}
echo "GPUNUM: "$gpunum
#save_model_name="colbert_generated_pretrain"
save_model_name="colbert_schedule_no_neg_schedule"
save_model_name="colbert_schedule_iter0"
save_model_name="colbert_schedule"
scoretemperature=1e-3
scoretemperature=2e-2
gradient_accumulation_steps=1
#save_model_name="colbert_schedule_""$scoretemperature""_decouple"
#save_model_name="colbert_schedule_partqww_""$scoretemperature""_decouple"
save_model_name="colbert_reader"
#save_model_name="colbert_schedule_2e-2_decouple"
#save_model_name="colbert_medqa_2e-2_noweight"
save_model_name="colbert_medqa_${scoretemperature}_weight"
#save_model_name="colbert_medqa_2e-2"
retriever_model="colbert_schedule_2e-2_decouple"
base_save_dir="/home2/awu/testcb/"
task="geo"
#output_dir=$base_save_dir/output/webq/$save_model_name
output_dir=$base_save_dir/output/$task/$save_model_name
output_dir=$base_save_dir/output/$task/"$save_model_name"
index_path=$base_save_dir/index/$task/$save_model_name/
server_index_path=$base_save_dir/index/$task/$save_model_name/

#corpus_path="/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"
corpus_path="/home/awu/experiments/geo/others/testcb/data/collection/all_paragraph_segmented.json"

run_fold(){
# batchsize = 44
#GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
#python -m colbert.train_cbqa_retrieval_gen \
GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" --master_port 47799 -m colbert.train_cbqa_retrieval_gen_medqa \
    --amp \
    --do_train \
    --batch_size 12 \
    --epoch 20 \
    --retriever_lr 3e-5 \
    --lr 3e-5 \
    --scoretemperature $scoretemperature \
    --train_files bm25_ir_sort-train-$1 \
    --dev_files bm25_ir_sort-dev-$1 \
    --test_files bm25_ir_sort-test-$1 \
    --output_dir $output_dir \
    --logging_steps 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps
}

#similarity=cosine
#--collection data/collection/all_paragraph_segmented.json \
#
#save_model_name="webq_colbert_t5_answer_title_pretrain"

pre_tensorize_corpus(){
  python -m colbert.training.pre_tokenize_corpus
}

index_model() {
#  pre_tensorize_corpus
#    CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node="$gpunum" -m  colbert.training.index_model \
  GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true \
    CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --master_port 47799 --nproc_per_node=$gpunum -m  colbert.training.index_model \
    --amp \
    --bsize 224 \
    --checkpoint $output_dir \
    --collection $corpus_path \
    --index_path $index_path \
    --sample 0.2
#    --partitions 2000
}

faiss_index_model(){
  CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.faiss_index_model \
    --amp \
    --bsize 224 \
    --collection $corpus_path \
    --checkpoint $output_dir \
    --index_path $index_path \
    --sample 0.05
#    --partitions 2000
}

start_retriever_server(){
  CUDA_VISIBLE_DEVICES=$server_gpu python -m colbert.training.model_helper_server \
    --index_path $index_path
}

#
eval_metric_for_model(){
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=$gpu \
  python -m colbert.train_cbqa_retrieval_gen_medqa \
    --amp \
    --do_eval \
    --checkpoint $output_dir/pytorch.bin \
    --batch_size 1 \
    --index_path $index_path/
}

train_reader(){
GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" -m colbert.train_cbqa_retrieval_gen_reader \
    --amp \
    --do_train \
    --batch_size 1 \
    --epoch 20 \
    --retriever_lr 3e-5 \
    --lr 3e-5 \
    --scoretemperature $scoretemperature \
    --train_files bm25_ir_sort-train-$1 \
    --dev_files bm25_ir_sort-dev-$1 \
    --test_files bm25_ir_sort-test-$1 \
    --output_dir $output_dir \
    --logging_steps 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps
}

for i in $(seq 0 0);do
#  sleep 14400
    echo $i
  if [ "$command" = "train" ];then
       run_fold $i
#       index_model
#       faiss_index_model
#       start_retriever_server
#    run_fold $i
  elif [ "$command" = "index" ];then
       index_model
       faiss_index_model
#       eval_metric_for_model $i
  elif [ "$command" = "faiss" ];then
       faiss_index_model
#    faiss_index_model $i
  elif [ "$command" = "server" ];then
       start_retriever_server
  elif [ "$command" = "eval" ];then
      eval_metric_for_model $i
  elif [ "$command" = "reader" ];then
      train_reader $i
  fi


done
