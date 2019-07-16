import torch
import torch.nn


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


def main():
    sizes = [(10, 10), (4, 4), (13, 11), (1, 1)]
    a2wa = AttentionToWordAlignment(6, 8)
    for nx, ny in sizes:
        inp = torch.rand(6, 8, nx, ny).unsqueeze(0)
        out = a2wa(inp)
        print(inp.shape, '->', out.shape)


if __name__ == '__main__':
    main()
