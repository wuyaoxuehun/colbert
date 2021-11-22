from torch import einsum
import torch


def score1(Q, D, q_mask=None, d_mask=None):
    if d_mask is not None:
        D = (D.permute(2, 0, 1) * d_mask).permute(1, 2, 0)
    Q = Q.unsqueeze(1)  # (Q, 1, seqQ, H)
    D = D.permute(0, 2, 1)  # (D, H, seqD)
    scores = (Q @ D).max(-1).values  # (Q, D, seqQ)
    if q_mask is not None:
        scores = scores.permute(1, 0, 2)  # (D, Q, seqQ)
        scores = (scores * q_mask).sum(-1).T
    else:
        scores = scores.sum(-1)
    return scores


def score2(Q, D, q_mask=None, d_mask=None):
    if d_mask is not None and q_mask is not None:
        D = D * d_mask[..., None]
        Q = Q * q_mask[..., None]
    scores = einsum("qmh,dnh->qdmn", Q, D).max(-1)[0].sum(-1)
    return scores


from time import time

if __name__ == '__main__':
    Q = torch.randn(8, 128, 768)
    D = torch.randn(18, 256, 768)
    q_mask = torch.randint(0, 2, (8, 128))
    d_mask = torch.randint(0, 2, (18, 256))
    # for fun in [score1, score2]:
    #     t1 = time()
    #     for i in range(200):
    #         fun(Q, D, q_mask, d_mask)
    #     print(time() - t1)
    print(torch.allclose(score1(Q, D, q_mask, d_mask), score2(Q, D, q_mask, d_mask)))

