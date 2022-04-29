import torch.nn.functional as F
import torch
from torch import einsum
from torch.nn import BCELoss


def listnet_loss(y_pred, y_true, eps=1e-10, *args, **kwargs):
    P_y_i = F.softmax(y_true, dim=-1)
    P_z_i = F.softmax(y_pred, dim=-1) + eps
    # print(P_y_i[0], P_z_i[0])
    return torch.mean(- torch.sum(P_y_i * torch.log(P_z_i), dim=-1))
    # return torch.mean(- torch.sum(P_y_i * P_z_i, dim=-1))


def kl_loss(y_pred, y_true):
    return F.kl_div(F.log_softmax(y_pred, dim=-1), F.softmax(y_true, dim=-1), reduction='batchmean')


def binary_listnet(y_pred, y_true):
    softmax_preds = torch.log_softmax(y_pred, dim=1)
    normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
    normalizer[normalizer == 0.0] = 1.0
    normalizer = normalizer.expand(-1, y_true.shape[1])
    y_true = torch.div(y_true, normalizer)
    loss = torch.mean(-torch.sum(y_true * softmax_preds, dim=1))
    return loss


def BiEncoderNllLoss(
        scores,
        positive_idx_per_question,
        hard_negative_idx_per_question=None,
        dual=False
):
    """
    Computes nll loss for the given lists of question and ctx vectors.
    Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
    loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
    :return: a tuple of loss value and amount of correct predictions per batch
    """
    # eps = 0
    # print(torch.isfinite(scores))
    # print((scores != scores).any())
    softmax_scores = F.log_softmax(scores, dim=1)
    # print(scores[0])
    # print(softmax_scores[0])

    loss = F.nll_loss(
        softmax_scores,
        torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        reduction="mean",
    )
    return loss
    # input(loss)
    if not dual:
        return loss
    dual_scores = scores[:, ::2].T
    dual_softmax_scores = F.log_softmax(dual_scores, dim=1)
    dual_loss = F.nll_loss(
        dual_softmax_scores,
        torch.arange(0, dual_softmax_scores.size(0), dtype=torch.long).to(softmax_scores.device),
        reduction="mean",
    )
    # max_score, max_idxs = torch.max(softmax_scores, 1)
    # correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

    # if loss_scale:
    #     loss.mul_(loss_scale)
    lam = 1
    # lam = 1

    return lam * loss + (1 - lam) * dual_loss
    # return loss, correct_predictions_count


def BiEncoderNllLossTri(
        scores,
        positive_idx_per_question,
        hard_negative_idx_per_question=None,
        dual=False
):
    """
    Computes nll loss for the given lists of question and ctx vectors.
    Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
    loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
    :return: a tuple of loss value and amount of correct predictions per batch
    """
    softmax_scores = F.log_softmax(scores, dim=1)
    loss = F.nll_loss(
        softmax_scores,
        torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        reduction="mean",
    )
    # return loss

    dual_scores = scores[:, ::2].T
    dual_softmax_scores = F.log_softmax(dual_scores, dim=1)
    dual_loss = F.nll_loss(
        dual_softmax_scores,
        torch.arange(0, dual_softmax_scores.size(0), dtype=torch.long).to(softmax_scores.device),
        reduction="mean",
    )
    # pair_scores =
    # pair_loss =
    # max_score, max_idxs = torch.max(softmax_scores, 1)
    # correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

    # if loss_scale:
    #     loss.mul_(loss_scale)
    # lam = 0.5
    lam = 1

    # return lam * loss + (1 - lam) * dual_loss
    return lam * loss + 0.1 * dual_loss
    # return loss, correct_predictions_count


def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    # mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    # preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    # observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1))


def with_ordinals(y, n, padded_value_indicator=-1):
    """
    Helper function for ordinal loss, transforming input labels to ordinal values.
    :param y: labels, shape [batch_size, slate_length]
    :param n: number of ordinals
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: ordinals, shape [batch_size, slate_length, n]
    """
    dev = y.device
    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=dev)
    unsqueezed = y.unsqueeze(2).repeat(1, 1, n)
    mask = unsqueezed == padded_value_indicator
    ordinals = (unsqueezed >= one_to_n).type(torch.float)
    ordinals[mask] = padded_value_indicator
    return ordinals


def ordinal(y_pred, y_true, n, padded_value_indicator=-1):
    """
    Ordinal loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length, n]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param n: number of ordinal values, int
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device

    y_pred = y_pred.clone()
    y_true = with_ordinals(y_true.clone(), n)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output


def lambdaLoss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, weighing_scheme=None, k=None, sigma=1., mu=10.,
               reduction="sum", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()
    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args):
    return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def listMLEWeighted(y_pred, y_true, eps=1e-10, neg_weight_mask=None, decouple=True):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    # print(y_pred.size(), y_true.size())
    # input()
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    # mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    # preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    if neg_weight_mask is not None:
        # assert observation_loss.size() == neg_weight_mask.size()
        neg_weight_mask_shuffled = neg_weight_mask[:, random_indices]
        neg_weight_mask_sorted_by_true = torch.gather(neg_weight_mask_shuffled, dim=1, index=indices)
        if decouple:
            observation_loss = torch.log(cumsums[..., 1:] + eps) - preds_sorted_by_true_minus_max[..., :-1]
            observation_loss = observation_loss * neg_weight_mask_sorted_by_true[..., :-1]
        else:
            observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
            observation_loss = observation_loss * neg_weight_mask_sorted_by_true
    else:
        if decouple:
            observation_loss = torch.log(cumsums[..., 1:] + eps) - preds_sorted_by_true_minus_max[..., :-1]
        else:
            observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

        # print(neg_weight_mask_sorted_by_true)
        # print(observation_loss)
        # input()
        # from torch.distributed import get_rank
        # if get_rank() == 0:
        #     print(neg_weight_mask_sorted_by_true)
        #     print(observation_loss)
        # input()
        # observation_loss *= neg_weight_mask_sorted_by_true
        # print(neg_weight_mask.size(), y_pred.size(), y_true.size())
        # input()
    # print(observation_loss)
    # input()
    # observation_loss[mask] = 0.0
    # print(y_true[0].bool().sum())
    # return torch.mean(torch.sum(observation_loss, dim=1)) / (y_true[0].bool().sum())
    return torch.mean(torch.sum(observation_loss, dim=1))
    # return torch.mean(torch.sum(observation_loss, dim=1) * 2)


def listMLEPLWeighted(y_pred, y_true, eps=1e-10, neg_weight_mask=None):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    # random_indices = torch.randperm(y_pred.shape[-1])
    random_indices = _pl_sample(y_true, T=1)
    y_pred_shuffled = y_pred.gather(dim=1, index=random_indices)
    # y_true_shuffled = y_pred.gather(dim=1, index=random_indices)

    # y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
    #
    # # mask = y_true_sorted == padded_value_indicator
    #
    # preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    # preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = y_pred_shuffled.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = y_pred_shuffled - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp(), dim=1)
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    if neg_weight_mask is not None:
        # assert observation_loss.size() == neg_weight_mask.size()
        neg_weight_mask_shuffled = neg_weight_mask.gather(dim=1, index=random_indices)
        # neg_weight_mask_sorted_by_true = torch.gather(neg_weight_mask_shuffled, dim=1, index=indices)
        observation_loss *= neg_weight_mask_shuffled

    # observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1)) / y_true.size(1)


def _pl_sample(t, T=0.5):
    t /= T
    probs = F.softmax(t, dim=-1)
    random_indices = probs.multinomial(num_samples=t.size(1), replacement=False)
    return random_indices


import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def testAWL():
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
    loss1 = -100
    loss2 = 1
    loss_sum = awl(loss1, loss2)
    print(loss_sum)


def testlistloss():
    # y_i = torch.tensor([[0.9, 0.9, 0.8, 0.8, 0.8] + [0.6, 0.6, 0.6, 0.6, 0.6] * 7], dtype=torch.float)
    # z_i = torch.tensor([[1, 1, 1, 1, 1] + [0, 0, 0, 0, 0] * 7], dtype=torch.float)
    y_i = torch.tensor([[1, 0.6, 0.99, 0.2, 0.2]]) / 0.1
    z_i = torch.tensor([[1, 0.8, 0.5, 0.1, 0.06]])
    # t = torch.tensor([1])
    # loss = listnet_loss(y_i, z_i)
    # print(loss)
    # loss = binary_listnet(y_i, z_i)
    # from torch.nn import CrossEntropyLoss
    # print(loss)
    # print(CrossEntropyLoss()(y_i, t))
    loss = listMLE(y_i, z_i)
    print(loss)


def test_ordinal_loss():
    y_true = torch.tensor([[1., 2, 3]])
    y_pred = torch.tensor([[1., 2, 4]])
    print(lambdaLoss(y_pred=y_pred, y_true=y_true))


def BiEncoderNllLossMS(
        Q,
        D,
        q_mask,
        d_mask,
        positive_idx_per_question
):
    Q_norm, D_norm = [F.normalize(_, p=2, dim=-1) for _ in [Q, D]]
    D = D_norm * d_mask[..., None]
    Q = Q_norm * q_mask[..., None]
    scores_match = einsum("qmh,dnh->qdmn", Q, D).max(-1)[0]
    scores = scores_match.sum(-1)
    scores = scores / (q_mask.bool().sum(-1)[:, None])
    # scores = F.relu(einsum("qmh,dnh->qdmn", Q, D)).max(-1)[0].sum(-1)
    softmax_scores = F.log_softmax(scores / 2e-2, dim=1)
    loss1 = F.nll_loss(
        softmax_scores,
        torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        reduction="mean",
    )
    from colbert.training.pa_losses import MultiSimilarityLossSM
    d_sim_mat = einsum("qmh,dnh->qdmn", D, D).max(-1)[0].sum(-1)
    d_sim_mat = d_sim_mat / (d_mask.bool().sum(-1)[:, None])
    ms_loss = MultiSimilarityLossSM()
    t = 8 * 4
    trange = torch.arange(t) + 1
    labels = torch.stack([trange, trange, torch.zeros(t)]).to(device=Q.device).T.flatten()
    loss2 = ms_loss(d_sim_mat, labels)

    return loss1 + loss2
    # input(loss)
    if not dual:
        return loss
    dual_scores = scores[:, ::2].T
    dual_softmax_scores = F.log_softmax(dual_scores, dim=1)
    dual_loss = F.nll_loss(
        dual_softmax_scores,
        torch.arange(0, dual_softmax_scores.size(0), dtype=torch.long).to(softmax_scores.device),
        reduction="mean",
    )
    # max_score, max_idxs = torch.max(softmax_scores, 1)
    # correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

    # if loss_scale:
    #     loss.mul_(loss_scale)
    lam = 1
    # lam = 1

    return lam * loss + (1 - lam) * dual_loss
    # return loss, correct_predictions_count


if __name__ == '__main__':
    # testAWL()
    testlistloss()
    # test_ordinal_loss()
