import matplotlib.pyplot as plt
import sys
import torch


def main():
    if len(sys.argv) != 2:
        print('Usage: %s model' % sys.argv[0])
        sys.exit(1)

    nlayers = 6
    nheads = 8

    model = torch.load(sys.argv[1], map_location='cpu')
    weights = model['conv.weight']
    kx, ky = weights.shape[-2:]
    weights = weights.view(nlayers, nheads, kx, ky)

    for layer in range(nlayers):
        for head in range(nheads):
            ax = plt.subplot(nlayers, nheads, layer * nheads + head + 1)
            ax.set_axis_off()
            ax.autoscale_view()
            ax.imshow(weights[layer, head, :, :].numpy(), cmap='coolwarm', interpolation='nearest')

    plt.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.01, 0.01)
    plt.show()


if __name__ == '__main__':
    main()