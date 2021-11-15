
from colbert.training.training import train
from colbert.utils.parser import Arguments
from colbert.utils.runs import Run


def main():
    parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()

    args = parser.parse()

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    # args.lazy = args.collection is not None
    args.lazy = False

    with Run.context(consider_failed_if_interrupted=False):
        train(args)


if __name__ == "__main__":
    import cProfile
    cProfile.run("main()", "result")
    import pstats
    # pstats.Stats('result').sort_stats(-1).print_stats()
    # pstats.Stats('result').sort_stats(-1).print_stats()
