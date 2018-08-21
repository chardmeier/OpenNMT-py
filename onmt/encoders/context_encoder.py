import onmt
import torch


class SimpleGate(torch.nn.Module):
    def __init__(self, dim):
        super(SimpleGate, self).__init__()
        self.gate = torch.nn.Linear(2 * dim, dim, bias=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, in1, in2):
        z = self.sig(self.gate(torch.cat((in1, in2), dim=1)))
        return z * in1 + (1.0 - z) * in2


class ContextTransformerLayer(torch.nn.Module):
    """
    A single layer of the context transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, d_model, heads, d_ff, dropout):
        super(ContextTransformerLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.attn_gate = SimpleGate(d_model)

        self.feed_forward = onmt.modules.position_ffn.PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, context, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        attn_context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        ctx_context, _ = self.context_attn(context, context, input_norm)
        gated_context = self.attn_gate(attn_context, ctx_context)
        out = self.dropout(gated_context) + inputs
        return self.feed_forward(out)


class ContextTransformerEncoder(onmt.encoders.encoder.EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(ContextTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = torch.nn.ModuleList(
            [onmt.encoders.transformer.TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers - 1)])
        self.context_layer = ContextTransformerLayer(d_model, heads, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)

    def forward(self, inp, lengths=None):
        self._check_args(inp['src'], lengths)

        emb = self.embeddings(inp['src'])

        out = emb.transpose(0, 1).contiguous()
        words = inp['src'][:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)

        for i in range(self.num_layers - 1):
            out = self.transformer[i](out, mask)

        out = self.context_layer(out, inp['context'])

        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous()
