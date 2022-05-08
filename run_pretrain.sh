
gpu='1,2,3,4'
IFS=','
read -ra arr <<< "$gpu"
gpunum=${#arr[@]}
echo "GPUNUM: "$gpunum


GIT_PYTHON_REFRESH=quiet OMP_NUM_THREADS=1   TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" -m dense_pretrain.run_pretrain
#python -m dense_pretrain.run_pretrain