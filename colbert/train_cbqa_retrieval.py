from colbert.training.training_cbqa_retrieval import *
from colbert.utils.cbqa_parser import CBQAArguments


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


if __name__ == "__main__":
    # import line_profiler
    # import sys
    # profile = line_profiler.LineProfiler(main)  # 把函数传递到性能分析器
    # profile.enable()  # 开始分析
    # main()
    # profile.disable()  # 停止分析
    # profile.print_stats(sys.stdout)  # 打印出性能分析结果
    main()
    # import cProfile
    #
    # cProfile.run("main()", "result")
    # import pstats
    # pstats.Stats('result').sort_stats(-1).print_stats()
