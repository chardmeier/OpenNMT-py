import onmt.inputters.text_dataset
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print('Usage: %s in.pt out.pt' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    examples = torch.load(infile)
    for ex in examples:
        ex.src = [ex.src[0]]

    examples.sort_key = onmt.inputters.text_dataset.text_sort_key
    torch.save(examples, outfile)


if __name__ == '__main__':
    main()
