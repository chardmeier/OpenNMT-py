import attn_to_wa
import itertools
import sys
import torch


def load_alignment(f):
    return set(tuple(int(x) for x in ap.split('-')) for line in f for ap in line.rstrip('\n').split(' '))


def make_alignment(mat):
    hardmat = torch.gt(mat, 0)
    return set((i, j) for i in range(hardmat.shape[0]) for j in range(hardmat.shape[1]) if mat[i, j])


def moses_alignment(alig):
    return ' '.join('%d-%d' % ap for ap in sorted(alig))


def main():
    if len(sys.argv) not in (3, 4):
        print('Usage: %s model attn [gold]' % sys.argv[0])
        sys.exit(1)

    model_file = sys.argv[1]
    attn_file = sys.argv[2]
    gold_file = sys.argv[3] if len(sys.argv) == 4 else None

    model = torch.load(model_file, map_location='cpu')
    attn = torch.load(attn_file)

    if len(sys.argv) == 4:
        with open(gold_file, 'r') as f:
            gold = load_alignment(f)

    nlayers = 6
    nheads = 8

    a2wa = attn_to_wa.AttentionToWordAlignment(nlayers, nheads)
    a2wa.load_state_dict(model)

    match_count = 0
    pred_count = 0
    gold_count = 0

    for at, g in itertools.zip_longest(attn, gold):
        pred = a2wa(at)
        alig = make_alignment(pred)
        print(moses_alignment(alig))

        if g:
            match_count += len(alig.intersection(g))
            pred_count += len(alig)
            gold_count += len(g)

    if gold:
        aer = 1.0 - match_count / (pred_count + gold_count)
        print('Alignment error rate: %g' % aer, file=sys.stderr)


if __name__ == '__main__':
    main()
