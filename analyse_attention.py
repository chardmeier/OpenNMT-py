import sys
import torch


def main():
    if len(sys.argv) != 3:
        print('Usage: %s attention_log sntno' % sys.argv[0], file=sys.stderr)
        sys.exit(1)
    logfile = sys.argv[1]
    sntno = int(sys.argv[2])

    src_text, context_docs, gate_vals, attn_mt, attn_coref, chain_map = torch.load(logfile, map_location='cpu')

    s_src = src_text[sntno]
    s_context_doc = context_docs[sntno]
    s_gate = gate_vals[sntno]
    s_attn_mt = attn_mt[sntno]
    s_attn_ctx = attn_coref[chain_map == sntno]

    print('Sentence %d' % sntno)
    print('%d words' % len(s_src))
    print('%d active chains' % s_attn_ctx.shape[0])
    print(s_src)

    return


if __name__ == '__main__':
    main()

