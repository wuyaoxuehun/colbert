#!/bin/bash

command=$1
gpu='6,7'
gpu='4'
server_gpu="1"
#gpu='2'
IFS=','
read -ra arr <<< "$gpu"
gpunum=${#arr[@]}
echo "GPUNUM: "$gpunum

scoretemperature=1e-3
scoretemperature=5e-3
gradient_accumulation_steps=1

#save_model_name="colbert_reader_1"
save_model_name="colbert_reader_1_rank"
#save_model_name="colbert_schedule_2e-2_decouple"
#retriever_model="colbert_schedule_2e-2_decouple"
retriever_model="colbert_schedule"
base_save_dir="/home2/awu/testcb/"
task="geo"
#output_dir=$base_save_dir/output/webq/$save_model_name
output_dir=$base_save_dir/output/$task/$save_model_name
output_dir=$base_save_dir/output/$task/"$save_model_name"
index_path=$base_save_dir/index/$task/$save_model_name/
server_index_path=$base_save_dir/index/$task/$retriever_model/

#corpus_path="/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"
corpus_path="/home/awu/experiments/geo/others/testcb/data/collection/all_paragraph_segmented.json"

run_fold(){
# batchsize = 44
#GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
#python -m colbert.train_cbqa_retrieval_gen \
GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" -m colbert.train_cbqa_retrieval_gen \
    --amp \
    --do_train \
    --batch_size 2 \
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
    CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$gpunum -m  colbert.training.index_model \
    --amp \
    --bsize 384 \
    --checkpoint $output_dir \
    --collection $corpus_path \
    --index_path $index_path \
    --sample 0.2
#    --partitions 2000
}
#
faiss_index_model(){
  CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.faiss_index_model \
    --amp \
    --bsize 512 \
    --collection $corpus_path \
    --checkpoint $output_dir \
    --index_path $index_path \
    --sample 0.2
#    --partitions 2000
}

start_retriever_server(){
  CUDA_VISIBLE_DEVICES=$server_gpu python -m colbert.training.model_helper_server \
    --index_path $index_path
}

start_retriever_server_reader(){
   CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES='4' python -m colbert.training.model_helper_server \
    --index_path $server_index_path
}

#
eval_metric_for_model(){
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="1" \
  python -m colbert.train_cbqa_retrieval_gen \
    --amp \
    --do_eval \
    --checkpoint $output_dir/pytorch.bin \
    --batch_size 1 \
    --index_path $server_index_path/
}

train_reader(){
#GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
#python -m torch.distributed.launch --nproc_per_node="$gpunum" -m colbert.train_cbqa_retrieval_gen_reader \
GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="1,4" \
python -m torch.distributed.launch --nproc_per_node=2 -m colbert.train_cbqa_retrieval_gen_reader \
    --amp \
    --do_train \
    --batch_size 2 \
    --epoch 6 \
    --retriever_lr 3e-5 \
    --lr 3e-5 \
    --scoretemperature $scoretemperature \
    --train_files bm25_ir_sort-train-$1 \
    --dev_files bm25_ir_sort-dev-$1 \
    --test_files bm25_ir_sort-test-$1 \
    --output_dir $output_dir \
    --logging_steps 1 \
    --gradient_accumulation_steps 1
}

eval_metric_for_reader(){
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="1" \
  python -m colbert.train_cbqa_retrieval_gen_reader \
    --amp \
    --do_eval \
    --checkpoint $output_dir/pytorch.bin \
    --batch_size 1 \
    --index_path $server_index_path/
}

for i in $(seq 0 0);do
    echo $i
  if [ "$command" = "train" ];then
       run_fold $i
       index_model
       faiss_index_model
#       start_retriever_server
#    run_fold $i
  elif [ "$command" = "index" ];then
       index_model
       faiss_index_model
  elif [ "$command" = "faiss" ];then
       faiss_index_model
#    faiss_index_model $i
  elif [ "$command" = "server" ];then
       start_retriever_server
  elif [ "$command" = "eval" ];then
      eval_metric_for_model $i
  elif [ "$command" = "reader" ];then
      train_reader $i
    elif [ "$command" = "eval_reader" ];then
    eval_metric_for_reader $i
  elif [ "$command" = "server_reader" ];then
       start_retriever_server_reader
  fi


done