import argparse
import random


def random_pair():
    src = 'abcdefghijklmnopqrstuvwxyz'
    tgt = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    i = random.randint(0, len(src) - 1)
    return src[i], tgt[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', required=True, help='File name to output source text to.')
    parser.add_argument('-tgt', required=True, help='File name to output target text to.')
    parser.add_argument('-docids', required=True, help='File name to output document IDs to.')
    parser.add_argument('-size', type=int, default=1000, help='Number of documents to generate.')
    parser.add_argument('-max_clusters', type=int, default=10, help='Number of documents to generate.')
    parser.add_argument('-tok_before', type=int, default=10, help='Tokens before coref mention.')
    parser.add_argument('-tok_after', type=int, default=10, help='Tokens after coref mention.')
    opt = parser.parse_args()

    with open(opt.src, 'w') as f_src, open(opt.tgt, 'w') as f_tgt, open(opt.docids, 'w') as f_docids:
        for i in range(opt.size):
            docid = 'doc%d' % i
            cluster_id = random.randint(0, opt.max_clusters - 1)

            txt = [random_pair() for _ in range(opt.tok_before)]
            txt.append(('antecedent%d' % cluster_id, 'ANTECEDENT%d' % cluster_id))
            txt += [random_pair() for _ in range(opt.tok_after)]

            src = ' '.join(p[0] for p in txt)
            tgt = ' '.join(p[1] for p in txt)
            print(src, file=f_src)
            print(tgt, file=f_tgt)
            print(docid, file=f_docids)

            txt = [random_pair() for _ in range(opt.tok_before)]
            txt.append(('anaphor', 'ANAPHOR%d' % cluster_id))
            txt += [random_pair() for _ in range(opt.tok_after)]

            src = ' '.join(p[0] for p in txt)
            tgt = ' '.join(p[1] for p in txt)
            print(src, file=f_src)
            print(tgt, file=f_tgt)
            print(docid, file=f_docids)


if __name__ == '__main__':
    main()
