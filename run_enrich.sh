
gpu='0'
IFS=','
read -ra arr <<< "$gpu"
gpunum=${#arr[@]}
echo "GPUNUM: "$gpunum


GIT_PYTHON_REFRESH=quiet OMP_NUM_THREADS=1   TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" -m retrieval_enrich.enrich_trainer
#python -m dense_pretrain.run_pretrain