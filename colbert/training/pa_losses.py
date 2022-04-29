import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes=self.nb_classes, embedding_size=self.sz_embed, softmax_scale=self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings=False)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg=None):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        # self.margin = 0.0

        # self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        # self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))
        # input(sim_mat)
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            if labels[i] == 0:
                continue
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
            # print(pos_pair_, neg_pair_)
            # input()
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


class MultiSimilarityLossSM(nn.Module):
    def __init__(self, cfg=None):
        super(MultiSimilarityLossSM, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        # self.margin = 0.0

        # self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        # self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, sim_mat, labels):
        feats = sim_mat
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        # sim_mat = torch.matmul(feats, torch.t(feats))
        # input(sim_mat)
        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            if labels[i] == 0:
                continue
            pos_pair_ = sim_mat[i][labels == labels[i]]
            # pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            # print(sim_mat[i])
            # print(pos_pair_, max(sim_mat[i]) - epsilon)
            # input()
            # print(pos_pair_ < max(sim_mat[i]) - epsilon)
            # input()
            pos_pair_ = pos_pair_[pos_pair_ < max(sim_mat[i]) - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
            # print(pos_pair_, neg_pair_)
            # input()
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            # if len(neg_pair) < 1 or len(pos_pair) < 1:
            #     continue

            # weighting step
            cur_loss = torch.tensor(0.).to(sim_mat.device)
            if len(pos_pair) >= 1:
                pos_loss = 1.0 / self.scale_pos * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                cur_loss += pos_loss
            if len(neg_pair) >= 1:
                neg_loss = 1.0 / self.scale_neg * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                cur_loss += neg_loss
            loss.append(cur_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss


def test():
    torch.manual_seed(6)
    feats = torch.randn(8, 3)
    feats = F.normalize(feats, p=2, dim=-1)
    feats = feats @ feats.T
    labels = torch.tensor([1, 0, 1, 0] * 2)
    ms_loss = MultiSimilarityLossSM()
    loss = ms_loss(feats, labels)
    print(loss)


if __name__ == '__main__':
    test()
