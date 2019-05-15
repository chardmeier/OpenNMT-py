import torch

from onmt.inputters.datareader_base import DataReaderBase


class ArtificialCorefDataReader(DataReaderBase):
    def __init__(self):
        super(ArtificialCorefDataReader, self).__init__()

    @classmethod
    def from_opt(cls, opt):
        return cls()

    def read(self, sequences, side, _dir=None):
        cluster_id = None
        cluster_emb = None
        text = [snt.decode('utf-8').rstrip('\n').split('\t')[1] for snt in sequences]
        for i, s in enumerate(snt.split(' ') for snt in text):
            coref_per_snt = []
            pos_in_cluster = 0
            for j, w in enumerate(s):
                if w.startswith('antecedent'):
                    cluster_id = int(w[10:])
                    cluster_emb = torch.zeros(2, 1220)
                    cluster_emb[0, cluster_id] = 1
                    coref_per_snt.append(([((j, j), pos_in_cluster)], cluster_emb, cluster_id))
                    pos_in_cluster += 1
                elif w == 'anaphor':
                    coref_per_snt.append(([((j, j), pos_in_cluster)], cluster_emb, cluster_id))
                    pos_in_cluster += 1

            yield {side: (s, coref_per_snt), 'indices': i}

