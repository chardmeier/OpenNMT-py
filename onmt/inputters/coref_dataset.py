# -*- coding: utf-8 -*-
"""Define word-based embedders."""

import collections
import itertools
import six
import spacy
import traceback

import torch
import torchtext

import allennlp.data
import allennlp.models

from onmt.inputters.datareader_base import DataReaderBase
from onmt.utils.logging import logger


class CorefField(torchtext.data.RawField):
    vocab_cls = torchtext.vocab.Vocab
    sequential = True

    span_emb_size = 1220

    def __init__(self, *args, **kwargs):
        super(CorefField, self).__init__()

        self.max_mentions_before = kwargs.pop('max_mentions_before', 1000)
        self.max_mentions_after = kwargs.pop('max_mentions_after', 1000)

        self.name = kwargs.get('base_name')
        self.src_field = torchtext.data.Field(init_token=kwargs.get('bos'), eos_token=kwargs.get('eos'),
                                              pad_token=kwargs.get('pad'),
                                              include_lengths=kwargs.get('include_lengths'))
        self.unk_token = self.src_field.unk_token
        self.pad_token = self.src_field.pad_token
        self.init_token = self.src_field.init_token
        self.eos_token = self.src_field.eos_token
        self.vocab = None

    def __iter__(self):
        # this is to be compatible with a single-entry TextMultiField
        yield self.name, self.src_field

    @property
    def vocab(self):
        return self.src_field.vocab

    @vocab.setter
    def vocab(self, value):
        self.src_field.vocab = value

    def preprocess(self, example):
        src, coref = example
        return self.src_field.preprocess(src), coref

    def process(self, batch, device=-1):
        """
        Convert batch of examples into tensor format.

        :param batch (`torchtext.data.Batch`): input data
        :param device (`torch.device`): device to create tensors on
        :return: (`tuple(tuple(tensor, tensor), CorefContext`)
            The first element of the outer tuple is exactly the representation that OpenNMT would use for
            data_type 'text', that is, a pair of a padded matrix with word indices and the associated
            original sentence lengths. The second element contains the information relevant to the
            coref-mt system, see class `CorefContext`.
        """
        src_batch = self.src_field.process([x[0] for x in batch], device=device)
        if self.src_field.include_lengths:
            src_batch, lengths = src_batch
        else:
            lengths = None

        pad_len = src_batch.shape[0]

        l_chain_map = []
        l_chain_start = []
        l_span_embeddings = []
        l_mask = []
        l_mention_pos_in_chain = []

        total_chains = 0
        for i, ex in enumerate(batch):
            total_chains += len(ex[1])
            for spans, emb in ex[1]:
                chain_length = emb.shape[0]
                min_pos_in_cluster = min(s[1] for s in spans)
                max_pos_in_cluster = max(s[1] for s in spans)
                emb_from = max(0, min_pos_in_cluster - self.max_mentions_before)
                emb_to = min(chain_length, max_pos_in_cluster + self.max_mentions_after)

                l_chain_map.append(i)
                l_chain_start.append(min_pos_in_cluster)
                l_span_embeddings.append(emb[emb_from:emb_to, :])
                snt_mask = torch.zeros(pad_len, dtype=torch.uint8)
                snt_mention_pos_in_chain = torch.full((pad_len,), -1, device=device, dtype=torch.long)
                for span, pos_in_chain in spans:
                    snt_mask[span[0]:span[1] + 1] = 1
                    snt_mention_pos_in_chain[span[0]:span[1] + 1] = pos_in_chain
                l_mask.append(snt_mask)
                l_mention_pos_in_chain.append(snt_mention_pos_in_chain)

        if total_chains == 0:
            coref_context = None
        else:
            max_chain_length = max(emb.shape[0] for emb in l_span_embeddings)
            chain_map = torch.tensor(l_chain_map, device=device, dtype=torch.long)
            chain_start = torch.tensor(l_chain_start, device=device, dtype=torch.long)
            mention_pos_in_chain = torch.stack(l_mention_pos_in_chain, 0)
            span_embeddings = torch.zeros(total_chains, max_chain_length, self.span_emb_size, device=device)
            attention_mask = torch.zeros(total_chains, pad_len, max_chain_length, device=device, dtype=torch.uint8)

            for i, (emb, snt_mask) in enumerate(zip(l_span_embeddings, l_mask)):
                span_embeddings[i, :emb.shape[0], :] = emb
                attention_mask[i, snt_mask, emb.shape[0]:] = 1

            coref_context = CorefContext(chain_map, chain_start, span_embeddings,
                                         attention_mask, mention_pos_in_chain)

        # The unsqueeze is because we pretend to be a single-factor multi-field
        out_batch = src_batch.unsqueeze(-1), coref_context

        if self.src_field.include_lengths:
            return out_batch, lengths
        else:
            return out_batch


class CorefContext:
    def __init__(self, chain_map, chain_start, span_embeddings, attention_mask, mention_pos_in_chain):
        # A [nchains] long vector mapping chains to examples in the batch
        self.chain_map = chain_map
        # A [nchains] long vector indicating the chain-internal position of the first recorded mention
        self.chain_start = chain_start
        # A [nchains x max_chain_length x span_embedding_size] float tensor holding the mention embeddings
        # of all chains
        self.span_embeddings = span_embeddings
        # A [nchains x sentence_length x max_chain_length] byte tensor indicating the positions eligible for
        # attention (i.e., words belonging to a mention and the actual chain length)
        # Note: It's ZERO for eligible positions, one for those that can't receive attention.
        self.attention_mask = attention_mask
        # A [nchains x sentence_length] long tensor indicating the chain-internal position of each mention
        # in the sentence
        self.mention_pos_in_chain = mention_pos_in_chain


def coref_sort_key(ex):
    """ Sort using length of source sentences. """
    # Default to a balanced sort, prioritizing tgt len match.
    # TODO: make this configurable.
    if hasattr(ex, "tgt"):
        return len(ex.src), len(ex.tgt)
    return len(ex.src)


class CorefDataReader(DataReaderBase):
    def __init__(self, src_lang, tgt_lang, run_coref):
        logger.info('Loading Spacy model for %s' % src_lang)
        spacy_src = spacy.load(src_lang, disable=['parser', 'tagger', 'ner'])
        logger.info('Loading Spacy model for %s' % tgt_lang)
        spacy_tgt = spacy.load(tgt_lang, disable=['parser', 'tagger', 'ner'])
        self.spacy = {'src': spacy_src, 'tgt': spacy_tgt}
        self.doc_builder = DocumentBuilder(run_coref)

    @classmethod
    def from_opt(cls, opt):
        return cls(opt.src_lang, opt.tgt_lang, opt.run_coref)

    def read(self, sequences, side, _dir=None):
        """Read coref data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with CorefDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)

        if side != 'src':
            yield from self._read_not_src(sequences, side)
        else:
            yield from self._read_src(sequences)

    def _read_not_src(self, sequences, side):
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")

            tok = [t.text for t in self.spacy[side](seq)]
            yield {side: tok, 'indices': i}

    def _read_src(self, sequences):
        for b_docid, doc_in in itertools.groupby((l.split(b'\t', maxsplit=1) for l in sequences), key=lambda t: t[0]):
            tok_src = [[t.text for t in self.spacy['src'](snt.decode('utf-8').rstrip('\n'))] for _, snt in doc_in]
            docid = b_docid.decode('utf-8')
            logger.info('Document %s: %d segments' % (docid, len(tok_src)))

            try:
                doc = self.doc_builder.make_document(docid, tok_src)
            except Exception as err:
                # AllenNLP sometimes fails on weird data (e.g., single-sentence docs without any mentions)
                logger.error('Document creation failed. Skipping document.')
                traceback.print_exc()
                continue

            for i, s in enumerate(tok_src):
                yield {'src': (s, doc.coref_per_snt[i]), 'indices': i}


class DocumentBuilder:
    def __init__(self, coref_path, max_span_width=10):
        if coref_path is None:
            self.dataset_reader = allennlp.data.DatasetReader.by_name('coref')(max_span_width)
            self.coref_model = None
        else:
            cuda_device = 0 if torch.cuda.is_available() else -1
            logger.info('Loading coref model from %s (cuda_device=%d).' % (coref_path, cuda_device))
            coref_model = allennlp.models.load_archive(coref_path, cuda_device=cuda_device)

            config = coref_model.config.duplicate()
            dataset_reader_params = config['dataset_reader']
            self.dataset_reader = allennlp.data.DatasetReader.from_params(dataset_reader_params)

            self.coref_model = coref_model.model
            self.coref_model.eval()

    def make_document(self, docid, tok_src):
        instance = self.dataset_reader.text_to_instance(tok_src)
        if self.coref_model is not None:
            coref_pred = self.coref_model.forward_on_instance(instance)
            top_spans = coref_pred['top_spans']
            top_span_embeddings = torch.from_numpy(coref_pred['top_span_embeddings'])
            clusters = coref_pred['clusters']
            top_span_dict = {tuple(top_spans[i, :]): i for i in range(top_spans.shape[0])}
            cluster_embeddings = [torch.stack([top_span_embeddings[top_span_dict[s], :] for s in cl])
                                  for cl in clusters]
            spans_in_cluster = list(sorted(s for c in clusters for s in c))
            span_to_cluster = {s: (i, j) for i, c in enumerate(clusters) for j, s in enumerate(c)}
            snt_start = 0
            coref_per_snt = []
            for snt in tok_src:
                active_clusters = collections.defaultdict(list)
                snt_end = snt_start + len(snt)
                while spans_in_cluster and snt_start <= spans_in_cluster[0][0] < snt_end:
                    span = spans_in_cluster.pop(0)
                    cluster_id, pos_in_cluster = span_to_cluster[span]
                    active_clusters[cluster_id].append((tuple(pos - snt_start for pos in span), pos_in_cluster))
                coref_list = []
                for cluster_id, spans in active_clusters.items():
                    coref_list.append((spans, cluster_embeddings[cluster_id]))
                coref_per_snt.append(coref_list)
                snt_start = snt_end
        else:
            coref_per_snt = None

        return Document(docid, instance, coref_per_snt)


class Document:
    def __init__(self, docid, instance, coref_per_snt):
        self.docid = docid
        self.instance = instance
        self.coref_per_snt = coref_per_snt


def coref_fields(**kwargs):
    """Create coref fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        CorefField
    """
    return CorefField(**kwargs)

