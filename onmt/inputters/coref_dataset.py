# -*- coding: utf-8 -*-
"""Define word-based embedders."""

import collections
import itertools
import spacy

import torch
import torchtext

import allennlp.data

from onmt.inputters.dataset_base import (DatasetBase, PAD_WORD, BOS_WORD, EOS_WORD)


class CorefField(torchtext.data.RawField):
    vocab_cls = torchtext.vocab.Vocab
    sequential = True

    def __init__(self, *args, **kwargs):
        super(CorefField, self).__init__()
        self.src_field = torchtext.data.Field(*args, **kwargs)
        self.unk_token = self.src_field.unk_token
        self.pad_token = self.src_field.pad_token
        self.init_token = self.src_field.init_token
        self.eos_token = self.src_field.eos_token
        self.vocab = None
        self.span_emb_size = 1220

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
        src_batch = self.src_field.process([x[0] for x in batch], device=device)

        pad_len = src_batch[0].shape[0]

        total_chains = sum((len(x[1]) for x in batch if len(x[1]) > 0))
        max_chain_length = max((y[1].shape[0] for x in batch for y in x[1]), default=0)
        mask = torch.zeros(total_chains, pad_len, max_chain_length, device=device, dtype=torch.uint8)
        span_embeddings = torch.empty(total_chains, max_chain_length, self.span_emb_size, device=device)
        chain_map = torch.zeros(total_chains, device=device, dtype=torch.long)
        k = 0
        for i, ex in enumerate(batch):
            for spans, emb in ex[1]:
                chain_map[k] = i
                span_embeddings[k, :emb.shape[0], :] = emb
                for span in spans:
                    mask[k, span[0]:span[1] + 1, :emb.shape[0]] = 1
                k += 1

        return src_batch, (chain_map, span_embeddings, mask)


class CorefDataset(DatasetBase):
    """ Dataset for data_type=='coref'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    n_src_feats = 0
    n_tgt_feats = 0

    def __init__(self, examples, fields, filter_pred):
        super(CorefDataset, self).__init__(examples, fields, filter_pred)
        self.data_type = 'coref'

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        raise NotImplementedError

    @staticmethod
    def get_fields():
        """
        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = CorefField(pad_token=PAD_WORD, include_lengths=True)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        return 0


class DocumentBuilder:
    def __init__(self, coref_model, max_span_width=10):
        if coref_model is None:
            self.dataset_reader = allennlp.data.DatasetReader.by_name('coref')(max_span_width)
            self.coref_model = None
        else:
            config = coref_model.config.duplicate()
            dataset_reader_params = config['dataset_reader']
            self.dataset_reader = allennlp.data.DatasetReader.from_params(dataset_reader_params)

            self.coref_model = coref_model.model
            self.coref_model.eval()

    def make_document(self, docid, tok_src, tok_tgt):
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
            span_to_cluster = {s: i for i, c in enumerate(clusters) for s in c}
            snt_start = 0
            coref_per_snt = []
            for snt in tok_src:
                active_clusters = collections.defaultdict(list)
                snt_end = snt_start + len(snt)
                while spans_in_cluster and snt_start <= spans_in_cluster[0][0] < snt_end:
                    span = spans_in_cluster.pop(0)
                    active_clusters[span_to_cluster[span]].append(tuple(pos - snt_start for pos in span))
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


def create_coref_datasets(src_iter, tgt_iter, docid_iter, shard_size,
                          use_filter_pred=True, src_seq_length=50, tgt_seq_length=50,
                          src_lang='en', tgt_lang='fr', run_coref=None):
    spacy_src = spacy.load(src_lang, disable=['parser', 'tagger', 'ner'])
    spacy_tgt = spacy.load(tgt_lang, disable=['parser', 'tagger', 'ner'])

    current_shard_size = 0
    index_in_shard = 0
    examples = []

    fields = CorefDataset.get_fields()
    doc_builder = DocumentBuilder(run_coref)

    def filter_pred(example):
        """ ? """
        return 0 < len(example.src) <= src_seq_length and 0 < len(example.tgt) <= tgt_seq_length

    filter_pred = filter_pred if use_filter_pred else lambda x: True

    for docid, doc_in in itertools.groupby(zip(docid_iter, src_iter, tgt_iter), key=lambda t: t[0]):
        l_doc_in = list(doc_in)
        tok_src = [[t.text for t in spacy_src(snt_src.rstrip('\n'))] for _, snt_src, _ in l_doc_in]
        tok_tgt = [[t.text for t in spacy_tgt(snt_tgt.rstrip('\n'))] for _, _, snt_tgt in l_doc_in]
        doc = doc_builder.make_document(docid.rstrip('\n'), tok_src, tok_tgt)

        for i, (s, t) in enumerate(zip(tok_src, tok_tgt)):
            ex = torchtext.data.Example()
            ex.src = fields['src'].preprocess((s, doc.coref_per_snt[i]))
            ex.tgt = fields['tgt'].preprocess(t)
            ex.indices = fields['indices'].preprocess(index_in_shard)
            ex.docid = doc.docid
            ex.snt_in_doc = i
            examples.append(ex)
            index_in_shard += 1

        current_shard_size += sum(len(snt_src) - 1 for _, snt_src, _ in l_doc_in)

        if current_shard_size >= shard_size:
            yield CorefDataset(examples, fields, filter_pred)
            current_shard_size = 0
            index_in_shard = 0
            examples = []

    if examples:
        yield CorefDataset(examples, fields, filter_pred)


