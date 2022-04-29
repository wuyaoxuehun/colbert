import os

gpu = '4,0'
gpu = '2,3,4'
gpunum = len(gpu.split(","))
# for SCORE_TEMPERATURE in ["1e-2", "1e-3", "1e-4", "1e-5"]:
for SCORE_TEMPERATURE in ["1e-3", "2e-3", "3e-3", "1e-4", "3e-4", "5e-4", "1e-5", "3e-5", "5e-5"]:
    # for SCORE_TEMPERATURE in ["5e-3"]:
    # for SCORE_TEMPERATURE in ["2e-2"]:
    script = f"""
command=$1
gpu={gpu}
gpunum={gpunum} 
echo "GPUNUM: "$gpunum
#save_model_name="colbert_generated_pretrain"
save_model_name="colbert_schedule_no_neg_schedule"
save_model_name="colbert_schedule_iter0"
save_model_name="colbert_schedule_partqww_{SCORE_TEMPERATURE}"
#save_model_name="colbert_schedule_partqww"
base_save_dir="/home2/awu/testcb/"

task="geo"
#output_dir=$base_save_dir/output/webq/$save_model_name
output_dir=$base_save_dir/output/$task/$save_model_name
output_dir=$base_save_dir/output/$task/"$save_model_name"
index_path=$base_save_dir/index/$task/$save_model_name/

#corpus_path="/home2/awu/testcb/tests/webqdata/webq_corpus_word.json"
# corpus_path="/home/awu/experiments/geo/others/testcb/data/collection/all_paragraph_segmented.json"

#GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
#python -m colbert.train_cbqa_retrieval_gen

GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES="$gpu" \
python -m torch.distributed.launch --nproc_per_node="$gpunum" -m colbert.train_cbqa_retrieval_gen \
    --amp \
    --do_train \
    --batch_size 2 \
    --epoch 20 \
    --retriever_lr 3e-5 \
    --lr 3e-5 \
    --scoretemperature {SCORE_TEMPERATURE} \
    --train_files bm25_ir_sort-train-$1 \
    --dev_files bm25_ir_sort-dev-$1 \
    --test_files bm25_ir_sort-test-$1 \
    --output_dir $output_dir \
    --logging_steps 1 \
    --gradient_accumulation_steps 1


  GIT_PYTHON_REFRESH=quiet CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true \
    CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --nproc_per_node=$gpunum -m  colbert.training.index_model \
    --amp \
    --bsize 384 \
    --checkpoint $output_dir \
    --collection "none" \
    --index_path $index_path \
    --sample 0.2


  CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.faiss_index_model \
    --amp \
    --bsize 512 \
    --collection "none" \
    --checkpoint $output_dir \
    --index_path $index_path \
    --sample 0.2
    
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=$gpu \
  python -m colbert.train_cbqa_retrieval_gen \
    --amp \
    --do_eval \
    --checkpoint $output_dir/pytorch.bin \
    --batch_size 1 \
    --index_path $index_path/

"""
    #   CUDA_VISIBLE_DEVICES=$gpu python -m colbert.training.model_helper_server \
    #     --index_path $index_path
    # }
    #
    # #
    # eval_metric_for_model(){
    #   OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=$gpu \
    #   python -m colbert.train_cbqa_retrieval_gen \
    #     --amp \
    #     --do_eval \
    #     --checkpoint $output_dir/pytorch.bin \
    #     --batch_size 1 \
    #     --index_path $index_path/
    # }
    # pre_tensorize_corpus()
    # {
    #     python - m
    # colbert.training.pre_tokenize_corpus
    # }
    # print(script)
    os.system(command=script)
