import sys


def main():
    if len(sys.argv) != 3:
        print('Usage: %s ref cand' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    ref_name = sys.argv[1]
    cand_name = sys.argv[2]

    matches = 0
    total = 0
    with open(ref_name, 'r') as ref, open(cand_name, 'r') as cand:
        for ref_line, cand_line in zip(ref, cand):
            rt = ref_line.rstrip('\n').split(' ')
            ct = cand_line.rstrip('\n').split(' ')
            ref_anaph = [t for t in rt if t.startswith('ANAPHOR')]
            cand_anaph = [t for t in ct if t.startswith('ANAPHOR')]
            assert len(ref_anaph) == len(cand_anaph)
            matches += sum(1 for r, c in zip(ref_anaph, cand_anaph) if r == c)
            total += len(ref_anaph)

    print('%d / %d = %.2f' % (matches, total, matches / total))


if __name__ == '__main__':
    main()

