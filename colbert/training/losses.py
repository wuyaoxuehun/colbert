import torch.nn.functional as F
import torch


def listnet_loss(y_pred, y_true, eps=1e-10, *args, **kwargs):
    P_y_i = F.softmax(y_true, dim=-1)
    P_z_i = F.softmax(y_pred, dim=-1) + eps
    # print(P_y_i[0], P_z_i[0])
    return torch.mean(- torch.sum(P_y_i * torch.log(P_z_i), dim=-1))
    # return torch.mean(- torch.sum(P_y_i * P_z_i, dim=-1))


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
        hard_negative_idx_per_question,
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
    # lam = 0.5
    lam = 1

    return lam * loss + (1 - lam) * dual_loss
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


def listMLEWeighted(y_pred, y_true, eps=1e-10, neg_weight_mask=None):
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
    if neg_weight_mask is not None:
        # assert observation_loss.size() == neg_weight_mask.size()
        neg_weight_mask_shuffled = neg_weight_mask[:, random_indices]
        neg_weight_mask_sorted_by_true = torch.gather(neg_weight_mask_shuffled, dim=1, index=indices)
        observation_loss *= neg_weight_mask_sorted_by_true

    # observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1)) / (y_true[0].bool().sum())


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
    y_i = torch.tensor([[1, 0.6, 0.99, 0.2, 0.2]])
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


if __name__ == '__main__':
    # testAWL()
    testlistloss()
