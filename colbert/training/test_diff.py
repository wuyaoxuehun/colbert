from colbert.modeling.colbert_list_qa_gen import ColBERT_List_qa
import torch
from transformers import BertForMaskedLM

from conf import pretrain


def test():
    colbert_qa = ColBERT_List_qa(load_old=False)
    colbert_qa.load("/home2/awu/testcb//output/geo/colbert_qdww_listnet_noschedule1/pytorch.bin")

    colbert_qa1 = ColBERT_List_qa(load_old=False)
    colbert_qa1.load("/home2/awu/testcb//output/geo/colbert_qdww_listnet_noschedule/pytorch.bin")

    print(colbert_qa1.linear.weight[0, :100])
    print(colbert_qa.linear.weight[0, :100])
    for i in range(768):

        if not (torch.allclose(colbert_qa1.linear.weight[i, :], colbert_qa.linear.weight[i, :])):
            print(colbert_qa1.linear.weight[i, :10])
            print(colbert_qa.linear.weight[i, :10])
            input(i)

def testetst():
    model = BertForMaskedLM.from_pretrained(pretrain)
    model1 = BertForMaskedLM.from_pretrained(pretrain)
    print(torch.allclose(model.cls.weight, model1,cls.weight))

if __name__ == '__main__':
    testetst()


