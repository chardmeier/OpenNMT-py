import spacy
import sys


def main():
    if len(sys.argv) != 2:
        print('Usage: %s spacy-model' % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    nlp = spacy.load(model)
    nlp.remove_pipe('tagger')
    nlp.remove_pipe('parser')
    nlp.remove_pipe('ner')

    for line in sys.stdin:
        snt = nlp(line.rstrip('\n'))
        toks = [t.text for t in snt]
        print(' '.join(toks))


if __name__ == '__main__':
    main()
