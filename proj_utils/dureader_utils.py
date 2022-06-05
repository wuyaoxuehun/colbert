from tqdm import tqdm


def csv_reader(input_file, delimiter='\t'):
    def gen():
        with open(input_file, 'r', encoding='utf8') as fd:
            for i, line in enumerate(fd):
                slots = line.rstrip('\n').split(delimiter)
                yield slots

    return gen()


def get_dureader_ori_corpus():
    medqa_dir = "/home2/awu/testcb/data/dureader/dureader-retrieval-baseline-dataset/passage-collection/"
    sents = []
    for i in range(0, 4):
        corpus_file = medqa_dir + f"part-0{i}"
        print(i)
        tsents = [_[2] for _ in csv_reader(corpus_file)]
        for sent in tqdm(tsents):
            sents.append(sent)
    return sents


if __name__ == '__main__':
    pass