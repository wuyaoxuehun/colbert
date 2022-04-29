import torch

# from torch.distributed import get_rank
# from random_utils import train_set_seed
from awutils.random_utils import train_set_seed

print(torch.cuda.device_count())
from colbert.utils.cbqa_parser import CBQAArguments
# from colbert.training.training_cbqa_retrieval_gen import train
from colbert.training.training_cbqa_retrieval_gen_medqa import *


def main():
    parser = CBQAArguments(description='trainig cbqa.')
    args = parser.parse()
    if args.do_train:
        train(args)
    if args.do_eval:
        # do_eval(args, mode="dev")
        eval_retrieval_for_model(args)
    if args.do_test:
        do_eval(args, mode="test")


# from colbert.modeling.colbert_list_qa_gen import ModelHelper
from colbert.indexing.faiss_indexers import *

if __name__ == "__main__":
    train_set_seed(1234)
    print(torch.cuda.device_count())
    main()
    exit()
    from line_profiler import LineProfiler
    import sys

    lp = LineProfiler()  # 把函数传递到性能分析器
    # lp.add_function(eval_retrieval_for_model)
    # for f in (ColBERT_List_qa.forward, ColBERT_List_qa.query, ColBERT_List_qa.doc,
    #           CostomTokenizer.tokenize_multiple_parts, CostomTokenizer.tokenize_q_noopt_segmented_dict, CostomTokenizer.tokenize_d_segmented_dict):
    # for f in [CostomTokenizer.tokenize_parts, train, CBQADataset.tokenize_for_train_retriever,
    #           CostomTokenizer.tokenize_q_noopt_segmented_dict, ColBERT_List_qa.forward, ColBERT_List_qa.query, ColBERT_List_qa.doc]:
    for f in [ColBERT_List_qa.forward, ColbertRetriever.search]:
        lp.add_function(f)
    # lp.add_function(ModelHelper.retrieve_for_encoded_queries)
    # lp.add_function(DPRRetriever.search)
    # lp.add_function(DenseFlatIndexer.search_knn)

    lp_wrapper = lp(main)
    lp_wrapper()
    # profile.disable()  # 停止分析
    lp.print_stats(sys.stdout)  # 打印出性能分析结果

    # main()
    # import cProfile
    #
    # cProfile.run("main()", "result")
    # import pstats
    # pstats.Stats('result').sort_stats(-1).print_stats()
