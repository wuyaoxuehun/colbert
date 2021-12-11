from colbert.utils.cbqa_parser import CBQAArguments
from colbert.training.training_cbqa_retrieval_gen import *


def main():
    parser = CBQAArguments(description='trainig cbqa.')

    args = parser.parse()

    # assert args.batch_size % args.gradient_accumulation_steps == 0, ((args.bsize, args.accumsteps),
    #                                                                  "The batch size must be divisible by the number of gradient accumulation steps.")

    if args.do_train:
        train(args)
    if args.do_eval:
        # do_eval(args, mode="dev")
        eval_retrieval_for_model(args)
    if args.do_test:
        do_eval(args, mode="test")


from colbert.modeling.colbert_list_qa_gen import ModelHelper
from colbert.indexing.faiss_indexers import DPRRetriever, DenseFlatIndexer
if __name__ == "__main__":
    # main()
    # exit()
    from line_profiler import LineProfiler
    import sys

    lp = LineProfiler()  # 把函数传递到性能分析器
    lp.add_function(eval_retrieval_for_model)
    lp.add_function(ModelHelper.retrieve_for_encoded_queries)
    lp.add_function(DPRRetriever.search)
    lp.add_function(DenseFlatIndexer.search_knn)

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
