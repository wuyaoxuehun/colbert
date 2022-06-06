#!/bin/bash
command=$1
gpu='0,2,3,4'
IFS=','
read -ra arr <<< "$gpu"
gpunum=${#arr[@]}
echo "GPUNUM: "$gpunum

train(){
GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" --master_port 47790 -m colbert.train_colbert \
    --output_dir ./temp \
    --report_to mlflow \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 28 \
    --num_train_epochs 20 \
    --fp16
}

index_model() {
  GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true \
  CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --master_port 47790 --nproc_per_node=$gpunum -m  colbert.training.index_model
}

faiss_index_model(){
  CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.faiss_index_model
}


server(){
  CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.model_helper_server server
}

evaluate(){
  CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.model_helper_server evaluate

}


if [ "$command" = "train" ];then
     train
elif [ "$command" = "index" ];then
     index_model
elif [ "$command" = "faiss" ];then
     faiss_index_model
elif [ "$command" = "server" ];then
     server
elif [ "$command" = "eval" ];then
    evalaluate
fi

