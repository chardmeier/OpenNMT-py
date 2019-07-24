import sys
import torch


def main():
    for fname in sys.argv[1:]:
        ds = torch.load(fname)
        for x in ds.examples:
            src = [w.encode('unicode_escape') for w in x.src]
            print(b' '.join(src).encode('ascii'), file=sys.stdout)

            tgt = [w.encode('unicode_escape') for w in x.tgt]
            print(b' '.join(tgt).encode('ascii'), file=sys.stderr)
