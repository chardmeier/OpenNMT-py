import allennlp.models
import argparse
import gzip
import torch

import onmt
from onmt.utils.logging import logger, init_logger
from onmt.inputters.coref_dataset import CorefDataset, create_coref_datasets


def openfile(fname, mode='r'):
    if fname.endswith('.gz'):
        return gzip.open(fname, mode, encoding='utf-8')
    else:
        return open(fname, mode)


def process_corpus(corpus_type, file_stem, src, tgt, docids, shard_size, run_coref=None):
    ds_files = []
    with openfile(src) as f_src, openfile(tgt) as f_tgt, openfile(docids) as f_docids:
        for index, dataset in enumerate(create_coref_datasets(f_src, f_tgt, f_docids, shard_size, run_coref=run_coref)):
            # We save fields in vocab.pt separately, so make it empty.
            dataset.fields = []

            pt_file = "{:s}.{:s}.{:d}.pt".format(file_stem, corpus_type, index)
            logger.info(" * saving %s data shard to %s." % (corpus_type, pt_file))
            torch.save(dataset, pt_file)
            ds_files.append(pt_file)
    return ds_files


def main():
    parser = argparse.ArgumentParser(description='Tokenise and preprocess corpus for coref-mt.')
    parser.add_argument('-train', nargs=3, help='Training corpus (src, tgt, docids).', required=True)
    parser.add_argument('-valid', nargs=3, help='Validation corpus (src, tgt, docids).', required=True)
    parser.add_argument('-shard_size', type=int, default=10 * 1024 * 1024, help='Shard size in bytes.')
    parser.add_argument('-save', help='Output file prefix.', required=True)
    parser.add_argument('-run_coref', help='Run coreference resolver during preprocessing. Takes model as parameter.')

    group = parser.add_argument_group('Vocab')
    group.add_argument('-src_vocab', default="",
                       help="""Path to an existing source vocabulary. Format:
                       one word per line.""")
    group.add_argument('-tgt_vocab', default="",
                       help="""Path to an existing target vocabulary. Format:
                       one word per line.""")
    group.add_argument('-src_vocab_size', type=int, default=50000,
                       help="Size of the source vocabulary")
    group.add_argument('-tgt_vocab_size', type=int, default=50000,
                       help="Size of the target vocabulary")

    group.add_argument('-src_words_min_frequency', type=int, default=0)
    group.add_argument('-tgt_words_min_frequency', type=int, default=0)

    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")
    args = parser.parse_args()

    init_logger()

    if args.run_coref:
        logger.info('Loading coref model from %s.' % args.run_coref)
        coref_model = allennlp.models.load_archive(args.run_coref)
    else:
        coref_model = None

    logger.info('Processing training corpus.')
    train_dataset_files = process_corpus('train', args.save, args.train[0], args.train[1], args.train[2],
                                         args.shard_size,
                                         run_coref=coref_model)

    logger.info('Processing validation corpus.')
    process_corpus('valid', args.save, args.valid[0], args.valid[1], args.valid[2], args.shard_size,
                   run_coref=coref_model)

    logger.info("Building & saving vocabulary...")
    fields = CorefDataset.get_fields()

    fields = onmt.inputters.build_vocab(train_dataset_files, fields, 'coref',
                                        args.share_vocab,
                                        args.src_vocab,
                                        args.src_vocab_size,
                                        args.src_words_min_frequency,
                                        args.tgt_vocab,
                                        args.tgt_vocab_size,
                                        args.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = args.save + '.vocab.pt'
    torch.save(onmt.inputters.save_fields_to_vocab(fields), vocab_file)


if __name__ == '__main__':
    main()
