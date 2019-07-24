import argparse
import sys
import torch
import torch.nn
import torch.optim


class AttentionToWordAlignment(torch.nn.Module):
    def __init__(self, nlayers, nheads, kernel_size=3):
        super(AttentionToWordAlignment, self).__init__()

        self.nlayers = nlayers
        self.nheads = nheads

        self.conv = torch.nn.Conv2d(nlayers * nheads, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, attn):
        nx, ny = attn.shape[3:]
        cnvin = attn.view(-1, self.nlayers * self.nheads, nx, ny)
        return self.conv(cnvin).squeeze(1)


def alignment_matrices(srcf, tgtf, aligf):
    for s, t, a in zip(srcf, tgtf, aligf):
        srclen = len(s.split(' '))
        tgtlen = len(t.split(' '))
        matrix = torch.zeros(srclen, tgtlen)
        for ap in a.rstrip('\n').split(' '):
            si, ti = (int(x) for x in ap.split('-'))
            matrix[si, ti] = 1
        yield matrix


def lazy_load_attention(fnames):
    for fn in fnames:
        attn = torch.load(fn)
        yield from iter(attn)
        del attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-attn', nargs='*', required=True, help='NMT attentions')
    parser.add_argument('-src', required=True, help='Tokenised source corpus')
    parser.add_argument('-tgt', required=True, help='Tokenised target corpus')
    parser.add_argument('-alig', required=True, help='Word alignments')
    parser.add_argument('-save_model', required=True, help='File name to save model to')
    parser.add_argument('-epochs', default=100, help='Number of epochs to train')
    parser.add_argument('-nlayers', type=int, default=6, help='Number of Transformer layers')
    parser.add_argument('-nheads', type=int, default=8, help='Number of attention heads per layer')
    parser.add_argument('-lr', type=float, default=.001, help='Learning rate')
    parser.add_argument('-momentum', type=float, default=.9, help='Momentum')
    args = parser.parse_args()

    a2wa = AttentionToWordAlignment(args.nlayers, args.nheads)
    optim = torch.optim.SGD(a2wa.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.BCEWithLogitsLoss()

    with open(args.src, 'r') as srcf, open(args.tgt, 'r') as tgtf, open(args.alig, 'r') as aligf:
        for epoch in range(args.epochs):
            running_loss = 0
            count = 0
            for attn_mat, alig_mat in zip(lazy_load_attention(args.attn), alignment_matrices(srcf, tgtf, aligf)):
                optim.zero_grad()
                pred = a2wa(attn_mat)
                loss = criterion(pred, alig_mat)
                loss.backward()
                optim.step()

                running_loss += loss.item()
                count += 1

            print('Epoch %d: Loss = %g' % (epoch, running_loss / count), file=sys.stderr)

    torch.save(a2wa.state_dict(), args.save_model)


if __name__ == '__main__':
    main()
