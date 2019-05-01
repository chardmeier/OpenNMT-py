import itertools
import random
import sys


def main():
    if len(sys.argv) != 7:
        print('Usage: %s src tgt docids src_out tgt_out docids_out' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    src, tgt, docids, src_out, tgt_out, docids_out = sys.argv[1:]

    print('Loading documents...', file=sys.stderr)
    with open(src, 'r') as f_src, open(tgt, 'r') as f_tgt, open(docids, 'r') as f_docids:
        docs = [list(lines) for docid, lines in itertools.groupby(zip(f_src, f_tgt, f_docids),
                                                                  lambda e: e[2].split('\t')[0])]

    print('Shuffling...', file=sys.stderr)
    random.shuffle(docs)

    print('Storing to disk...', file=sys.stderr)
    with open(src_out, 'w') as f_src, open(tgt_out, 'w') as f_tgt, open(docids_out, 'w') as f_docids:
        for lines in docs:
            for s, t, d in lines:
                f_src.write(s)
                f_tgt.write(t)
                f_docids.write(d)


if __name__ == '__main__':
    main()

