import torch

from onmt.inputters.datareader_base import DataReaderBase


class ArtificialCorefDataReader(DataReaderBase):
    def __init__(self, size):
        super(ArtificialCorefDataReader, self).__init__()
        self.size = size
        self.tok_before = 5
        self.tok_after = 5

    @classmethod
    def from_opt(cls, opt):
        return cls(opt.artificial_size)

    def read(self, sequences, side, _dir=None):
        for i, s in enumerate(snt.decode('utf-8').rstrip('\n').split(' ') for snt in sequences):
            coref_per_snt = []
            cluster_id = None
            cluster_emb = None
            for j, w in enumerate(s):
                if w.startswith('antecedent'):
                    cluster_id = int(w[10:])
                    cluster_emb = torch.zeros(2, 1220)
                    cluster_emb[0, cluster_id] = 1
                    coref_per_snt.append(((j, j), cluster_emb, cluster_id))
                elif w == 'anaphor':
                    coref_per_snt.append(((j, j), cluster_emb, cluster_id))

            yield {side: (s, coref_per_snt), 'indices': i}

