# -*- coding: utf-8 -*-
"""Define word-based embedders."""

import itertools
import spacy

import torch
import torchtext

import allennlp.data.dataset_readers

from onmt.inputters.dataset_base import (DatasetBase, PAD_WORD, BOS_WORD, EOS_WORD)


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

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

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
    def __init__(self, max_span_width=10):
        self.instance_builder = allennlp.data.dataset_readers.ConllCorefReader(max_span_width)

    def make_document(self, docid, tok_src, tok_tgt):
        return Document(docid, self.instance_builder.text_to_instance(tok_src))


class Document:
    def __init__(self, docid, instance):
        self.docid = docid
        self.instance = instance


def create_coref_datasets(src_iter, tgt_iter, docid_iter, shard_size,
                          use_filter_pred=True, src_seq_length=50, tgt_seq_length=50,
                          src_lang='en', tgt_lang='fr'):
    spacy_src = spacy.load(src_lang, disable=['parser', 'tagger', 'ner'])
    spacy_tgt = spacy.load(tgt_lang, disable=['parser', 'tagger', 'ner'])

    current_shard_size = 0
    index_in_shard = 0
    examples = []

    fields = CorefDataset.get_fields()
    doc_builder = DocumentBuilder()

    def filter_pred(example):
        """ ? """
        return 0 < len(example.src) <= src_seq_length and 0 < len(example.tgt) <= tgt_seq_length

    filter_pred = filter_pred if use_filter_pred else lambda x: True

    for docid, doc_in in itertools.groupby(zip(docid_iter, src_iter, tgt_iter), key=lambda t: t[0]):
        l_doc_in = list(doc_in)
        tok_src = [[t.text for t in spacy_src(snt_src.rstrip('\n'))] for _, snt_src, _ in l_doc_in]
        tok_tgt = [[t.text for t in spacy_tgt(snt_tgt.rstrip('\n'))] for _, _, snt_tgt in l_doc_in]
        doc = doc_builder.make_document(docid.rstrip('\n'), tok_src, tok_tgt)

        for s, t in zip(tok_src, tok_tgt):
            ex = torchtext.data.Example()
            ex.src = fields['src'].preprocess(s)
            ex.tgt = fields['tgt'].preprocess(t)
            ex.indices = fields['indices'].preprocess(index_in_shard)
            ex.doc = doc
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

