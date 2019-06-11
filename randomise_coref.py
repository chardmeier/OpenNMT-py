import sys
import torch


def load_sample(f_train, nsamples):
    ts = torch.load(f_train)
    vals = torch.empty(nsamples)
    i = 0
    for x in ts:
        for c in x.src[1]:
            j = min(c[1].nelement(), nsamples - i)
            vals[i:(i + j)] = torch.flatten(c[1])[:j]
            i += j
            if i >= nsamples:
                return vals


def main():
    if len(sys.argv) != 4:
        print('Usage: %s train.pt in.pt out.pt', file=sys.stderr)
        sys.exit(1)

    nsamples = 10000000

    f_train, f_in, f_out = sys.argv[1:]
    dist = load_sample(f_train, nsamples)

    testset = torch.load(f_in)
    for x in testset:
        for c in x.src[1]:
            n = c[1].nelement()
            s = torch.randint(nsamples, size=(n,))
            c[1].storage().copy_(dist[s].storage())

    torch.save(testset, f_out)


if __name__ == '__main__':
    main()
