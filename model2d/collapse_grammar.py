import sys
import re

if __name__ == '__main__':
    f = open(sys.argv[1])
    unique = {}

    for l in f:
        m = re.search(r'(\d+\.*\d*)\] \{(\d+)', l)

        production = l[:l.index('[')].strip()
        probability = float(m.group(1))
        count = int(m.group(2))

        if production in unique:
            p, c = unique[production]
            unique[production] = (p+probability, c+count)
        else:
            unique[production] = (probability, count)

    s = sorted(['%s [%f] {%d}' % (prod, prob, cnt) for prod,(prob,cnt) in unique.items()])

    for sent in s:
        print sent