import sys


def main():
    docids = {}
    for line in sys.stdin:
        s_docid = line.rstrip('\n')
        n_docid = docids.setdefault(s_docid, len(docids))
        print(n_docid)

    docid_list = [(v, k) for k, v in docids.items()]
    for no, name in sorted(docid_list):
        print(no, name, sep='\t')


if __name__ == '__main__':
    main()
