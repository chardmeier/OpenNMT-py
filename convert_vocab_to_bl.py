import onmt.inputters.text_dataset
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print('Usage: %s in.pt out.pt' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    voc = torch.load(infile)

    crfield = voc['src']
    blfield = onmt.inputters.text_dataset.text_fields(base_name='src', n_feats=0, include_lengths=True)
    blfield.base_field.vocab = crfield.vocab
    voc['src'] = blfield

    torch.save(voc, outfile)


if __name__ == '__main__':
    main()
