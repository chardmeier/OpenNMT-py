import sys
import torch


def main():
    for fname in sys.argv[1:]:
        ds = torch.load(fname)
        for x in ds.examples:
            src = [w.encode('unicode_escape') for w in x.src[0]]
            print(b' '.join(src).decode('ascii'), file=sys.stdout)

            tgt = [w.encode('unicode_escape') for w in x.tgt[0]]
            print(b' '.join(tgt).decode('ascii'), file=sys.stderr)


if __name__ == '__main__':
    main()