import torch
import torch.nn.functional as F
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            # attn = attn.masked_fill(mask == 0, -10000)
            attn = attn.masked_fill(mask == 0, -1e4)
        # print(attn)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class ScaledDotProductAttentionWithBias(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1, bias_weight=1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.bias_weight = bias_weight

    def forward(self, q, k, v, mask=None, bias=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn += bias * self.bias_weight
        if mask is not None:
            # attn = attn.masked_fill(mask == 0, -10000)
            attn = attn.masked_fill(mask == 0, -1e4)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, self.n_head * self.d_v)
        q = self.dropout(self.fc(q))
        # q += residual
        # q = self.layer_norm(q)
        return q


class MultiHeadAttentionWithBias(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, bias_weight=1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttentionWithBias(temperature=d_k ** 0.5, bias_weight=bias_weight)

        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, bias=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        if bias is not None:
            bias = bias.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask, bias=bias)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, self.n_head * self.d_v)
        q = self.dropout(self.fc(q))
        # q += residual
        # q = self.layer_norm(q)
        return q


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, hidden, mask=None):
        scores = self.score(hidden).squeeze(-1)
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e4)
        # print(scores)
        weights = F.softmax(scores, dim=-1)
        # weights = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.float)
        weights = weights.unsqueeze(1)

        weighted_sum = weights @ hidden
        weighted_sum = weighted_sum.squeeze(1)
        return weighted_sum


from torch.autograd import Variable


class CapsuleLayer_(nn.Module):
    def __init__(self, num_out, d_out, num_in, d_in, num_iterations=2):
        super(CapsuleLayer, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.num_iterations = num_iterations

        self.num_in = num_in
        self.num_out = num_out

        self.route_weights = nn.Parameter(torch.randn(num_out, d_in, d_out))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        B, I, IH = x.size()
        # priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
        # print(x.unsqueeze(1).size())
        # print(self.route_weights.repeat(B, 1, 1, 1).size())
        priors = (x.unsqueeze(1)) @ (self.route_weights.repeat(B, 1, 1, 1))  # B*O*I*OH
        # priors = priors.squeeze(1).view(B, self.num_out, self.d_out)

        logits = Variable(torch.zeros(B, self.num_out, self.num_in).to(device=x.device))
        for i in range(self.num_iterations):
            # print(logits)
            priors_t = priors.permute(0, 1, 3, 2)
            probs = F.softmax(logits, dim=1).unsqueeze(-1)
            print(probs.squeeze())
            # print(priors_t.size())
            # print(probs.size())
            v = (priors_t @ probs).squeeze(-1)  # B*O*OH
            # print(v.size())
            outputs = self.squash(v)

            if i != self.num_iterations - 1:
                delta_logits = (priors @ (outputs.unsqueeze(-1))).squeeze(-1)
                # logits = logits + delta_logits
                logits = delta_logits

        return outputs


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations=3):
        super().__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))

    @staticmethod
    def squash(tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
        logits = Variable(torch.zeros(*priors.size()).to(device=x.device))
        for i in range(self.num_iterations):
            probs = softmax(logits, dim=2)
            outputs = CapsuleLayer.squash((probs * priors).sum(dim=2, keepdim=True))

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits
                # logits = delta_logits

        return outputs.squeeze().transpose(0, 1)


def test_capsule():
    num_out, d_out, num_in, d_in = 2, 128, 3, 768
    # model = CapsuleLayer(num_out, d_out, num_in, d_in)
    model = CapsuleLayer_(num_capsules=2, num_route_nodes=3, in_channels=768, out_channels=128)
    # inputs = torch.randn(5, num_in, d_in)
    inputs = torch.randn(4, num_in, d_in)
    output = model(inputs)
    print(output.size())


def test_lstm():
    X = torch.randn(2, 4, 4)
    seq_len = torch.tensor([2, 3])
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch import nn
    from torch.autograd import Variable
    X_padded = pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
    lstm = nn.LSTM(4, 3, 1, batch_first=True)

    def init_hidden():
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, 2, 3)
        hidden_b = torch.randn(1, 2, 3)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    X, hidden = lstm(X_padded, init_hidden())
    X, l = pad_packed_sequence(X, batch_first=True)
    print(X.size())
    print(X)
    print(l)


def test_MHA():
    mha = MultiHeadAttentionWithBias(n_head=1, d_model=2, d_k=1, d_v=1, dropout=0)
    A = torch.randn(1, 4, 2)
    B = torch.randn(1, 4, 2)
    mask1 = torch.tensor([[1, 1, 1, 1]])
    mask2 = torch.tensor([[1, 1, 1, 0]])
    mask = mask1.unsqueeze(-1) @ mask2.unsqueeze(-2)
    bias = torch.randn(4, 4)
    print(A)
    print(B)
    print(mask)
    res = mha(q=A, k=B, v=B, mask=mask, bias=bias)
    print(res)


def test_MHA1():
    mha = nn.MultiheadAttention(2, 1)
    A = torch.randn(1, 4, 2)
    B = torch.randn(1, 4, 2)
    mask1 = torch.tensor([[1, 1, 0, 0]])
    mask2 = torch.tensor([[1, 1, 1, 0]])
    mask = mask1.unsqueeze(-1) @ mask2.unsqueeze(-2)
    mask = ~(mask.bool())
    mask2 = ~(mask2.bool())
    print(A)
    print(B)
    print(mask.bool())
    print(list(mask.size()), [1 * 1, A.size(0), B.size(0)])
    A = A.permute(1, 0, 2)
    B = B.permute(1, 0, 2)
    res = mha(query=A, key=B, value=B, key_padding_mask=mask2,
              need_weights=True, attn_mask=mask.bool())
    print(res[0])
    print(res[1])


def test_atn():
    batch = torch.randn(2, 4, 4)
    mask = torch.randint(0, 2, (2, 4))
    attention = Attention(d_model=4)
    print(batch)
    print(mask)
    res = attention(hidden=batch, mask=mask)
    print(res)


if __name__ == '__main__':
    # test_lstm()
    # test_capsule()
    test_MHA()
