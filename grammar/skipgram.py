import itertools as it
import skipgram as sg
import IPython

class SkipGram(object):
    def __init__(self, gram, skippedlist):
        self.gram = gram
        self.skippedlist = skippedlist

    def __hash__(self):
        return hash(tuple(self.gram))

    def __eq__(self, other):
        if other is ():
            return False
        return self.gram == other.gram

    def __len__(self):
        return len(self.gram)

    def __repr__(self):
        return 'SkipGram(%s, skipped=%s)' % (self.gram, self.skippedlist)


def kskipngrams(sentence,k,n):
    "Assumes the sentence is already tokenized into a list"
    if n == 0 or len(sentence) == 0:
        return None
    grams = []
    for i in range(len(sentence)-n+1):
        grams.extend(initial_kskipngrams(sentence[i:],k,n))
    return grams

def initial_kskipngrams(sentence,k,n):
    if n == 1:
        return [SkipGram([sentence[0]],[[]])]
    grams = []
    for j in range(min(k+1,len(sentence)-1)):
        kmjskipnm1grams = initial_kskipngrams(sentence[j+1:],k-j,n-1)
        if kmjskipnm1grams is not None:
            for gram in kmjskipnm1grams:
                grams.append(SkipGram([sentence[0]]+(['_'] if j>0 else [])+gram.gram,
                             [sentence[1:j+1]+gram.skippedlist[0]]))
    return grams

def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable( it.combinations(xs,n) for n in range(2,len(xs)+1) )

if __name__ == '__main__':
    # a = "the quick brown fox jumps over the lazy dog".split()
    # print kskipngrams(a,0,1)
    # IPython.embed()

    strings = ['on', 'near to', 'far from', 'to the left of', 'to the right of',
               'to the front of', 'to the back of']
    maxlen = 4
    k = maxlen
    substrings = {}
    for string in strings:
        phrase = string.split()
        # print phrase
        subs = []
        for n in range(1,maxlen+1):
            skipgrams = sg.kskipngrams(phrase,k,n)
            if skipgrams:
                subs.extend(skipgrams)
        # subs = map(tuple, subs)
        # subs = set(subs)
        substrings[string] = subs

    # for key, val in substrings.items():
    #     print key
    #     print val
    #     print
    # exit()

    commons = []
    for stringset in powerset(strings):
        str1 = stringset[0]
        sgs1 = substrings[str1]
        for str2 in stringset[1:]:
            sgs2 = substrings[str2]
            common = []
            for sg1 in sgs1:
                if sg1 in sgs2:
                    sg2 = sgs2[sgs2.index(sg1)]
                    common.append(SkipGram(sg1.gram,sg1.skippedlist+sg2.skippedlist))
            sgs1 = common
        common = sgs1
        # common = list(common)
        common.sort(reverse=True, key=lambda x: len(x))
        if len(common) > 0:
            longest = common[0]
        else:
            longest = ()
        commons.append((stringset,longest))

    # for stringset, longest in commons:
    #     print stringset
    #     print longest
    #     print
    # exit()

    new_commons = []
    for i, (stringset1, longest1) in enumerate(commons):
        stringset1 = set(stringset1)
        add = True
        for stringset2, longest2 in commons[i+1:]:
            if longest1 == longest2 and \
               stringset1.issubset(set(stringset2)):
               add = False
               break
        if add:
            new_commons.append((stringset1, longest1))
    commons = new_commons


    commons.sort(reverse=True, key=lambda x: (len(x[1]),len(x[0])))
    for common in commons[:10]:
        print common[0]
        print common[1]
        print 
