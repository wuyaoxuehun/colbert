import time

import math
import torch
from tqdm import tqdm
from transformers import Trainer
from torch import nn, einsum
import torch.nn.functional as F

from colbert.modeling.model_utils import span_mean

hidden_size = 768
proj_size = 128
import numpy as np

torch.manual_seed(1)
np.random.seed(1)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
steps = 2000
epoch_total = 10
dist_idx = [0, 1, 2]
dist = [0.1, 0.2, 0.7]
labels = []
for i in range(steps):
    labels.append(int(np.random.choice(dist_idx, p=dist)))


def annealing(cur_step, total, max_limit=15, min_limit=0.5):
    return max_limit - cur_step / (total - 1) * (max_limit - min_limit)


# for i in range(10):
#     print(annealing(i, total=10))
# input()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, proj_size)
        self.linear2 = nn.Linear(proj_size, 1)
        self.linear3 = nn.Linear(proj_size, proj_size)
        self.linear = nn.Linear(proj_size, 2)

    def forward(self, t: torch.Tensor, evalaluate=False, T=None):
        embd = self.linear1(t)
        scores = self.linear2(embd).squeeze(-1)
        proj = self.linear3(embd)

        max_idx = scores.argmax()
        selected = proj[max_idx]

        # if not evalaluate:
        #     selected_idx = F.gumbel_softmax(scores.view(1, 3), hard=True, tau=T)
        #     max_idx = selected_idx.argmax()
        #     selected = (selected_idx @ proj).view(1, proj_size)
        # else:
        #     max_idx = scores.argmax()
        #     selected = proj[max_idx]

        logits = self.linear(selected).squeeze()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.unsqueeze(0), (logits.argmax().unsqueeze(0)) if max_idx == np.random.choice(dist_idx, p=dist) else ((1 - logits.argmax()).unsqueeze(0)))
        return loss


def test_optimzer():
    from transformers import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    device = "cpu"
    model = Model().to(device)
    optimizer_grouped_parameters = [
        {
            "params": model.linear1.parameters(),
            # "lr": 1
        },
        {
            "params": model.linear2.parameters(),
            # "lr": 2
        },
        {
            "params": model.linear3.parameters(),
        },
        {
            "params": model.linear.parameters()
        }
    ]
    lr = 5e-3
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    def lr_lambda_fun(num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            # return lr
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return lr_lambda

    # from allennlp.training.learning_rate_schedulers.linear_with_warmup import LinearWithWarmup
    # sd = LinearWithWarmup(optimizer=optimizer, num_epochs=1, num_steps_per_epoch=10, warmup_steps=2, last_epoch=-1)

    # scheduler = LambdaLR(optimizer, [lr_lambda_fun(2, steps), lr_lambda_fun(2, steps), lr_lambda_fun(2, steps)])
    scheduler = LambdaLR(optimizer, [lr_lambda_fun(steps, steps * epoch_total)] * 4)
    lrs = [[] for _ in range(3)]

    dp = torch.randn((steps, 3, hidden_size))
    epoch_losses = []
    for epoch in tqdm(range(epoch_total)):
        losses = []
        T = annealing(epoch, epoch_total)
        for i in range(steps):
            optimizer.zero_grad()
            # a = torch.randn((3, 10), requires_grad=True)
            a = dp[i]
            loss = model(a.to(device), False, T)
            loss.backward()
            # for idx, group in enumerate(optimizer_grouped_parameters):
            #     lrs[idx].append(group['lr'])
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
        epoch_losses.append(sum(losses) / len(losses))
    model.eval()
    with torch.no_grad():
        losses = 0.
        for i in range(steps):
            a = dp[i]
            loss = model(a.to(device), True)
            losses += float(loss)
        print(losses / steps)

        # sd.step_batch(i+1)
    print(epoch_losses)
    print(lrs)


def testimdb():
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    print(dataset['train'][0])


def test_tokenizer():
    words = ["a", "dog", "is", "a", "duck"]
    mask = [1, 1, 0, 0, 1]
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("/home/awu/experiments/pretrain/bert_wwm_ext_c3")
    inputs = tokenizer.encode_plus(words, is_split_into_words=True, add_special_tokens=True,
                                   max_length=10, padding='max_length', truncation=True)
    print(inputs.word_ids())
    print(tokenizer.decode(inputs['input_ids']))


def testspanmax():
    sl, input_dim, span_size = 128, 768, 64
    bs = 16
    sequence_tensor = torch.randn(bs, sl, input_dim)
    span_start = torch.randint(0, sl - 5, (bs, span_size, 1))
    span_len = torch.randint(1, 5, (bs, span_size, 1))
    span_end = span_start + span_len
    span_indices = torch.cat([span_start, span_end], dim=-1)
    span_indices_mask = torch.randint(0, 2, (bs, span_size))
    from allennlp.modules.span_extractors import MaxPoolingSpanExtractor, EndpointSpanExtractor
    ext = MaxPoolingSpanExtractor(input_dim=input_dim)
    end_ext = EndpointSpanExtractor(input_dim=input_dim)
    from colbert.training.CBQADataset_gen_medqa import CBQADataset
    ds = CBQADataset(file_abbrs="medqa_filter_merge-test-0", doc_maxlen=256)
    bs = 16
    for t in [ext, end_ext, span_mean]:
        t1 = time.time()
        for i in tqdm(range(0, len(ds), bs)):
            (q_ids, q_attention_mask, q_active_spans, q_active_padding), *_ = ds.tokenize_for_train_retriever([ds[i] for i in range(i, i + bs)])
            sequence_tensor = torch.randn(bs * 2, q_ids.size(1), 768).cuda()
            if t is not span_mean:
                q_active_spans[:, :, 1] -= 1
            output = t(sequence_tensor, q_active_spans.cuda())
        print(output.size())
        print(time.time() - t1)


def test_acl():
    from bs4 import BeautifulSoup
    import requests
    # papers = "https://dblp.org/db/conf/acl/acl2021-1.html"
    papers = "https://dblp.org/db/conf/emnlp/emnlp2019-1.html"
    auts = []
    # for i in tqdm(range(12)):
    #     papers = f"https://dblp.org/db/conf/eccv/eccv2020-{i}.html"
    #     paper_content = requests.get(papers)
    #     paper_soup = BeautifulSoup(paper_content.content, "lxml")
    #     names = paper_soup.find_all("span", itemprop="name")
    #     for name in names:
    #         auts.append(str(name.text).strip())
    paper_content = requests.get(papers)
    paper_soup = BeautifulSoup(paper_content.content, "lxml")
    names = paper_soup.find_all("span", itemprop="name")
    for name in names:
        auts.append(str(name.text).strip())
    page = "https://www.cmu.edu/math/people/phd-students.html"
    content = requests.get(page)
    content.encoding = content.apparent_encoding
    # print(content.content[:100])
    soup = BeautifulSoup(content.content, "lxml")
    persons = []
    for p in soup.find_all("h3"):
        persons.append(str(p.text).strip())
    print(auts)
    print(set(auts) & set(persons))


def test_entropy():
    dim = 2
    torch.manual_seed(1)
    qlen, dlen = 8, 16
    Q = torch.randn(1, qlen, 2)
    D = torch.randn(1, dlen, 2)
    Q = F.normalize(Q, p=2, dim=-1)
    D = F.normalize(D, p=2, dim=-1)
    active_padding = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    # active_padding = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]])
    # word_weight = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1.]])
    word_weight = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2.]])
    word_weight[active_padding == 0] = -1e4
    if True:
        scale_factor = torch.log(active_padding.sum(-1)) / math.log(256) / math.sqrt(dim)
        # print(word_weight[0])
        word_weight = word_weight * scale_factor[:, None]
        print(scale_factor)
        # print(word_weight[0])
    softmax_word_weight = F.softmax(word_weight, dim=-1)
    print(softmax_word_weight)
    return
    # t_norm = t_norm * softmax_word_weight[..., None]
    Q = Q * softmax_word_weight[..., None]
    scores_match = einsum("qmh,dnh->qdmn", Q, D).max(-1)[0]
    scores = scores_match.sum(-1)
    print(scores)


def testhftrainer():
    from transformers import HfArgumentParser, TrainingArguments, trainer
    from proj_conf.training_arguments import MyTraniningArgs
    import sys
    sys.argv = ["program.py", "--output_dir=abc/def", "--learning_rate=1e-5"]
    parser = HfArgumentParser((MyTraniningArgs,))
    training_args, *_ = parser.parse_args_into_dataclasses()
    print(training_args)


if __name__ == '__main__':
    # test_optimzer()
    # test_tokenizer()
    # testimdb()
    # testspanmax()
    # test_acl()
    # test_entropy()
    testhftrainer()
