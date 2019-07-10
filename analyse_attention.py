import collections
import sys
import torch


Cluster = collections.namedtuple('Cluster', ['cluster_id', 'local_spans', 'spans', 'mentions', 'gate', 'attention'])


def main():
    if len(sys.argv) != 2:
        print('Usage: %s attention_log' % sys.argv[0], file=sys.stderr)
        sys.exit(1)
    logfile = sys.argv[1]

    src_text, context_docs, chain_id, gate_vals, attn_mt, attn_coref, chain_map =\
        torch.load(logfile, map_location='cpu')

    for sntno, s_src in enumerate(src_text):
        sntlen = len(s_src)
        s_context_doc = context_docs[sntno]
        s_gate = gate_vals[sntno, :sntlen, :]
        s_attn_mt = attn_mt[sntno]
        s_attn_ctx = attn_coref[chain_map == sntno]
        s_cluster_ids = chain_id[chain_map == sntno]

        print('\n\nSentence %d' % sntno)
        print('%d words' % len(s_src))
        print('%d active chains' % s_attn_ctx.shape[0])
        print(s_src)

        clusters = []
        for i, c in enumerate(s_cluster_ids):
            spans = s_context_doc.coref_pred['clusters'][c]
            local_spans = s_context_doc.coref_per_snt[sntno][i][0]
            mentions = [s_context_doc.coref_pred['document'][a:b + 1] for a, b in spans]
            attn = s_attn_ctx[i, :, :sntlen, :len(spans)]
            clusters.append(Cluster(i, local_spans, spans, mentions, s_gate, attn))

        for c in clusters:
            print('Cluster %d' % c.cluster_id)
            for h in range(c.attention.shape[0]):
                print('Attention head %d' % h)
                print(' ' * 52, end='')
                for (a, b), pos in c.local_spans:
                    print('   %10s' % str(s_gate[a:b + 1].mean(dim=-1).tolist()), end='')
                print()
                print(' ' * 52, end='')
                for (a, b), pos in c.local_spans:
                    print('   %10s' % str(s_src[a:b + 1]), end='')
                print()
                for i, (s, m) in enumerate(zip(c.spans, c.mentions)):
                    print('%10s  %40s' % (str(s), str(m)), end='')
                    for (a, b), pos in c.local_spans:
                        print('   %10s' % str(c.attention[h, a:b + 1, i].tolist()), end='')
                    print()
                print()

    return


if __name__ == '__main__':
    main()

