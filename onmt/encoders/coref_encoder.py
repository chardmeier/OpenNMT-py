import collections
import math
import onmt
import torch

from onmt.inputters.coref_dataset import CorefField
from onmt.encoders.encoder import EncoderBase


# The standard OpenNMT implementations of gates and positional embeddings do additional things we don't want,
# therefore we provide our own basic versions of them.

class MaskedGate(torch.nn.Module):
    def __init__(self, dim, gate_per_word):
        super(MaskedGate, self).__init__()
        gate_dim = 1 if gate_per_word else dim
        self.gate = torch.nn.Linear(2 * dim, gate_dim, bias=True)
        self.sig = torch.nn.Sigmoid()

    def forward(self, in1, in2, mask):
        z = self.sig(self.gate(torch.cat((in1, in2), dim=-1))).masked_fill(mask.unsqueeze(-1), 1.0)
        return z * in1 + (1.0 - z) * in2


# mostly copied from onmt.modules.embeddings.PositionalEncoding
class CorefPositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=1000):
        super(CorefPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
        self.dim = dim

    def forward(self, mode, emb, steps):
        """
        Add positional encoding to input vector. In mode 'chain', the input embeddings have dimensions
        `[chain_length x model_dim]` and steps is a 1D vector of dimension `[chain_length]`. In mode
        'query', the input embeddings have dimension `[nchains x sentence_length x model_dim]` and steps
        is a matrix of dimension `[nchains x sentence_length]`, containing for each word belonging to the
        chain its position in the chain, or -1 for words that aren't part of a chain.
        :param mode: (`str`) 'chain' or 'query'
        :param emb: (`FloatTensor`) Input embeddings
        :param steps: (`LongTensor`) Time steps to encode
        :return: the sum of `emb` and the positional embeddings (same dimension as `emb`)
        """
        if mode == 'chain':
            steps = steps.squeeze()
            assert steps.ndimension() == 1
            nchains = steps.shape[0]
            chain_length = emb.shape[1]
            pe = torch.empty((nchains, chain_length, self.dim), device=emb.device)
            for i in range(nchains):
                pe[i, :, :] = self.pe[steps[i]:steps[i] + chain_length]
        elif mode == 'query':
            assert steps.ndimension() == 2
            nitems = emb.shape[0]
            pe = torch.where(steps.unsqueeze(-1).expand_as(emb) >= 0,
                             torch.gather(self.pe.expand(nitems, -1, -1), 1,
                                          steps.unsqueeze(-1).expand(-1, -1, self.dim).clamp(min=0)),
                             torch.zeros_like(emb))
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

    def __init__(self, d_model, d_context, mt_heads, coref_heads, d_ff, dropout, coref_gate_per_word):
        super(CorefTransformerLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(mt_heads, d_model, dropout=dropout)
        self.linear_context = torch.nn.Linear(d_context, d_model, bias=True)
        self.context_attn = onmt.modules.MultiHeadedAttention(coref_heads, d_model, dropout=dropout)
        self.attn_gate = MaskedGate(d_model, coref_gate_per_word)
        self.positional_embeddings = CorefPositionalEncoding(d_model)

        self.feed_forward = onmt.modules.position_ffn.PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_attn = torch.nn.Dropout(dropout)
        self.dropout_ctx = torch.nn.Dropout(dropout)

    def forward(self, inputs, coref_context, mask):
        """
        Coref Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            coref_context (`CorefContext`): the context required for coref processing
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """

        # first compute standard self-attention
        input_norm = self.layer_norm(inputs)
        attn_context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)

        # Now the coref-specific parts.
        if coref_context is None:
            # document has no mentions
            gated_context = self.dropout_attn(attn_context)
        else:
            # # Linearly map span embeddings from the size used by AllenNLP to our model size.
            # emb_transformed = self.linear_context(coref_context.span_embeddings)
            emb_transformed = coref_context.coref_matrix
            # Multiply input rows so that we have one instance of the sentence for each chain referred to
            context_query = torch.index_select(input_norm, 0, coref_context.chain_map)
            # Add positional embeddings to span embeddings and query
            context_query = self.positional_embeddings('query', context_query, coref_context.mention_pos_in_chain)
            emb_transformed = self.positional_embeddings('chain', emb_transformed, coref_context.chain_start)
            # Attention to vectors in coref chain
            attention_mask = coref_context.attention_mask[:, :, :emb_transformed.shape[1]]
            ctx_out, _ = self.context_attn(emb_transformed, emb_transformed, context_query,
                                           mask=attention_mask, type='coref')
            # Reduce output so we get one row per example again
            ctx_context, sentence_mask = _aggregate_chains(input_norm.shape[0], ctx_out,
                                                           coref_context.chain_map, coref_context.attention_mask)

            # Gate to choose between coref attention and self-attention
            gated_context = self.attn_gate(self.dropout_attn(attn_context), self.dropout_ctx(ctx_context),
                                           sentence_mask)

        out = gated_context + inputs
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
           and a mask indicating by 0 the words in each sentence that are part of a coreference chain
    """
    sentence_mask = torch.ones(batch_size, ctx_out.shape[1], dtype=torch.uint8, device=ctx_out.device)
    out_list = [torch.full(ctx_out.shape[1:], float('-inf'), device=ctx_out.device)] * batch_size
    minus_inf = torch.tensor([float('-inf')], device=ctx_out.device)
    for i in range(ctx_out.shape[0]):
        ex_idx = chain_map[i]
        sentence_mask[ex_idx, :] = sentence_mask[ex_idx, :] * mask[i, :, 0]
        masked = torch.where(mask[i, :, 0].unsqueeze(1), minus_inf, ctx_out[i, :])
        out_list[ex_idx] = torch.max(out_list[ex_idx], masked)

    out = torch.stack(out_list, dim=0)
    # In the next line, torch.eq(...) is a workaround for the lack of torch.isinf in my version of torch
    return torch.where(torch.eq(out + 1, out), torch.zeros(1, device=ctx_out.device), out), sentence_mask


class CorefTransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, d_context, heads, coref_heads, d_ff,
                 dropout, embeddings, coref_gate_per_word):
        super(CorefTransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = torch.nn.ModuleList(
            [onmt.encoders.transformer.TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers - 1)])
        self.context_layer = CorefTransformerLayer(d_model, d_context, heads, coref_heads, d_ff,
                                                   dropout, coref_gate_per_word)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            CorefField.span_emb_size,
            opt.heads,
            opt.coref_heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.coref_gate_per_word)

    def create_encoder_memory(self):
        return CorefMemory(self.d_model)

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
        return emb, out.transpose(0, 1).contiguous(), lengths


class CorefMemory:
    def __init__(self, embedding_size):
        self.memory = {}
        self.embedding_size = embedding_size

    def store_batch(self, batch, outputs):
        for docid, doc_continues in zip(batch.docid, batch.doc_continues):
            if not doc_continues and docid in self.memory:
                del self.memory[docid]

        context = batch.src[0][1]
        if context is None:
            return

        for chain_id, idx in zip(context.chain_id, context.chain_map):
            docid = batch.docid[idx].item()
            chain_id = chain_id.item()
            if batch.doc_continues[idx]:
                doc_outputs = self.memory.setdefault(docid, {})
                chain_outputs = doc_outputs.setdefault(chain_id, [])
                chain_outputs.append(self._process_output(batch, outputs, idx))
                print('doc %d - chain %d: len %d' % (docid, chain_id, len(chain_outputs)))

    def _process_output(self, batch, outputs, idx):
        return torch.mean(outputs[:, idx, :].detach(), dim=0)

    def prepare_src(self, batch):
        inp, context = batch.src[0]

        if context is None:
            return inp, None

        chainlen_dim = 0
        for chain_id, idx in zip(context.chain_id, context.chain_map):
            docid = batch.docid[idx].item()
            chain_id = chain_id.item()
            if docid in self.memory and chain_id in self.memory[docid]:
                chainlen_dim = max(chainlen_dim, len(self.memory[docid][chain_id]))

        if chainlen_dim == 0:
            return inp, None

        coref_matrix = torch.empty(context.chain_id.shape[0], chainlen_dim, self.embedding_size)
        for i, (chain_id, idx) in enumerate(zip(context.chain_id, context.chain_map)):
            docid = batch.docid[idx].item()
            chain_id = chain_id.item()
            if docid in self.memory and chain_id in self.memory[docid]:
                chain = self.memory[docid][chain_id]
                coref_matrix[i, :len(chain), :] = torch.stack(chain, dim=0)

        context.coref_matrix = coref_matrix
        return inp, context
