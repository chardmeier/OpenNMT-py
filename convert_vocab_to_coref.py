import onmt.inputters.coref_dataset
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print('Usage: %s in.pt out.pt' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    voc = torch.load(infile)

    blvoc = voc['src']
    crvoc = onmt.inputters.coref_dataset.CorefField(base_name='src')
    crvoc.vocab = blvoc
    voc['src'] = crvoc

    torch.save(voc, outfile)


if __name__ == '__main__':
    main()
