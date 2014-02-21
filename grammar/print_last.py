# coding: utf-8
import shelve
f = shelve.open('rel4.shelf')
all_answers = f['all_answers']
f.close()
a = [val for tup in zip(*all_answers) for val in tup]
a.sort(reverse=True)
for t,c,s,sem,_ in a:
    print c
    print s
    print sem
    raw_input()
    