import math
import onmt
import torch


# The standard OpenNMT implementations of gates and positional embeddings do additional things we don't want,
# therefore we provide our own basic versions of them.

class SimpleGate(torch.nn.Module):
    def __init__(self, dim):
        super(SimpleGate, self).__init__()
        self.gate = torch.nn.Linear(2 * dim, dim, bias=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, in1, in2):
        z = self.sig(self.gate(torch.cat((in1, in2), dim=-1)))
        return z * in1 + (1.0 - z) * in2


# mostly copied from onmt.modules.embeddings.PositionalEncoding
class CorefPositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=200):
        super(CorefPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                              -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, emb, steps, mode):
        """
        Add positional encoding to input vector. In mode 'chain', the input embeddings have dimensions
        `[chain_length x model_dim]` and steps is a 1D vector of dimension `[chain_length]`. In mode
        'query', the input embeddings have dimension `[nchains x sentence_length x model_dim]` and steps
        is a matrix of dimension `[nchains x sentence_length]`, containing for each word belonging to the
        chain its position in the chain, or -1 for words that aren't part of a chain.
        :param emb: (`FloatTensor`) Input embeddings
        :param steps: (`LongTensor`) Time steps to encode
        :param mode: (`str`) 'chain' or 'query'
        :return: the sum of `emb` and the positional embeddings (same dimension as `emb`)
        """
        if mode == 'chain':
            steps = steps.squeeze()
            assert steps.ndimension() == 1
            chain_length = emb.shape[1]
            pe = self.pe[steps:steps + chain_length].unsqueeze(0)
        elif mode == 'query':
            assert steps.ndimension() == 2
            nitems = emb.shape[0]
            max_len = steps.shape[1]
            pe = torch.where(steps >= 0,
                             torch.gather(self.pe.expand(nitems, -1, -1), 1, steps.expand(-1, -1, self.dim)),
                             torch.zeros(nitems, max_len, self.dim))
        else:
            raise ValueError('Unknown mode %s, should be chain or query.' % mode)

        return emb + pe


class CorefTransformerLayer(torch.nn.Module):
    """
    A single layer of the coref transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_context, heads, d_ff, dropout):
        super(CorefTransformerLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.linear_context = torch.nn.Linear(d_context, d_model, bias=True)
        self.context_attn = onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.attn_gate = SimpleGate(d_model)
        self.positional_embeddings = CorefPositionalEncoding(self.d_model)

        self.feed_forward = onmt.modules.position_ffn.PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs, coref_context, mask):
        """
        Coref Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            context (`CorefContext`): the context required for coref processing
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """

        # first compute standard self-attention
        input_norm = self.layer_norm(inputs)
        attn_context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)

        # Now the coref-specific parts.
        # Linearly map span embeddings from the size used by AllenNLP to our model size.
        emb_transformed = self.linear_context(coref_context.span_embeddings)
        # Add positional embeddings to span embeddings and query
        emb_transformed = self.positional_embeddings(emb_transformed, pos_in_chain[:, 1:])
        context_query = self.positional_embeddings(emb_transformed, pos_in_chain[:, 0])
        # Multiply input rows so that we have one instance of the sentence for each chain referred to
        context_query = torch.index_select(context_query, 0, coref_context.chain_map)
        # Attention to vectors in coref chain
        ctx_out, _ = self.context_attn(emb_transformed, emb_transformed, context_query,
                                       mask=coref_context.attention_mask)
        # Reduce output so we get one row per example again
        ctx_context = _aggregate_chains(input_norm.shape[0], ctx_out,
                                        coref_context.chain_map, coref_context.chain_mask)

        # Gate to choose between coref attention and self-attention
        gated_context = self.attn_gate(attn_context, ctx_context)
        out = self.dropout(gated_context) + inputs
        return self.feed_forward(out)


def _aggregate_chains(batch_size, ctx_out, chain_map, mask):
    """
    Merges the output of the context attention, which has one entry per coreference chain, into a single
    entry per training example. For each word position that is part of a mention, we take the contribution
    of the corresponding chain. If a mention belongs to multiple chains, we do max-pooling over all of
    them.

    :param batch_size (`int`): Number of examples (sentences) in the batch.
    :param ctx_out (`FloatTensor`): Output of context attention
           `[total_chains x sentence_length x span_embedding_size]`
    :param chain_map (`LongTensor`): Mapping of coreference chains to examples `[total_chains]`
    :param mask (`ByteTensor`): Binary mask indicating which chains a word belongs to and which
           entries of ctx_out can be attended to.
           `[total_chains x sentence_length x max_chain_length]`
    :return: Per-sentence context attention matrix `[batch_size x sentence_length x span_embedding_size]`
    """
    out_list = [torch.full(ctx_out.shape[1:], float('-inf'), device=ctx_out.device)] * batch_size
    minus_inf = torch.tensor([float('-inf')], device=ctx_out.device)
    for i in range(ctx_out.shape[0]):
        ex_idx = chain_map[i]
        masked = torch.where(mask[i, :, 0].unsqueeze(1), ctx_out[i, :], minus_inf)
        out_list[ex_idx] = torch.max(out_list[ex_idx], masked)

    out = torch.stack(out_list, dim=0)
    # In the next line, torch.eq(...) is a workaround for the lack of torch.isinf in my version of torch
    return torch.where(torch.eq(out + 1, out), torch.zeros(1, device=ctx_out.device), out)


class CorefTransformerEncoder(onmt.encoders.encoder.EncoderBase):
    def __init__(self, num_layers, d_model, d_context, heads, d_ff, dropout, embeddings):
        super(CorefTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = torch.nn.ModuleList(
            [onmt.encoders.transformer.TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers - 1)])
        self.context_layer = CorefTransformerLayer(d_model, d_context, heads, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)

    def forward(self, inp, lengths=None):
        src, context = inp
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)

        for i in range(self.num_layers - 1):
            out = self.transformer[i](out, mask)

        out = self.context_layer(out, context, mask)

        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous()
