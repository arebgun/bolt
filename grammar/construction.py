import re
# import sempoles as sp
import common as cmn

class Property(object):
    pass

class Gradable(Property):
    pass

# class Measure(Property):
#     pass

class Applicable(Property):
    pass

class Relation(Property):
    pass

class Unknown(object):
    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.__class__.__name__ + ' "' + self.string + '"'

    def __str__(self):
        return self.__repr__()

    def prettyprint(self):
        return self.__repr__() + '\n'

class LexicalItem(object):
    def __init__(self, regex, sempole):
        self.regex = regex
        self._sempole = sempole

    def match(self, text):
        matches = re.finditer(self.regex, text)
        intervals = [cmn.Match(m.start(),m.end(), self, None) for m in matches]
        return intervals

    def sempole(self):
        return self._sempole.copy()

    def __repr__(self):
        return self.__class__.__name__ + ' "' + self.regex + '"'

    def prettyprint(self):
        return self.__repr__() + '\n'

    def collect_leaves(self):
        return [self.regex]

    def __hash__(self):
        return hash(self.prettyprint())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def applicabilities(self, context):
        potential_referent_scores = context.get_potential_referent_scores()
        return self.sempole.applicabilities(potential_referent_scores)


class Construction(object):
    
    def __init__(self, constituents):
        assert(len(constituents) == len(self.pattern))
        self.constituents = constituents

    @staticmethod
    def match_template(seq, pat):
        return (isinstance(seq, pat) or
                  (isinstance(seq, cmn.Match) and 
                    issubclass(seq.construction, pat)))

    @staticmethod
    def pattern_match(pattern, sequence):
        matches = []
        patlength = len(pattern)
        for i, item in enumerate(sequence[:len(sequence)-patlength+1]):
            if Construction.match_template(item, pattern[0]):
                if len(pattern) == len(sequence[i:i+patlength]):
                    all_match = True
                    for pat, seq in zip(pattern, sequence[i:i+patlength]):
                        # print seq, pat, isinstance(seq, pat)
                        all_match = (all_match and 
                                        Construction.match_template(seq, pat))
                    if all_match:
                        matches.append((i,i+patlength))
        return matches

    @classmethod
    def match(cls, sequence):
        matches = [cmn.Match(start, end, cls, sequence[start:end]) 
                   for start,end in cls.pattern_match(cls.pattern, sequence)]
        return matches

    @classmethod
    def partially_match(cls, sequence):
        '''Only supports single holes for now'''
        # print '    pattern:',pattern
        # print '    sequence:',sequence
        partial_matches = []
        # print cls.pattern
        # print '  sequence',sequence
        if len(cls.pattern) == 1:
            return partial_matches
            # for start in range(len(sequence)):
            #     for end in range(start+1, len(sequence)+1):
            #         print '    subsequence',sequence[start:end]
            #         hole = cmn.Hole(cls.pattern,
            #                         sequence[start:end])
            #         partial_match = cmn.Match(start=start,
            #                                   end=end,
            #                                   construction=cls,
            #                                   constituents=[hole])
            #         partial_matches.append(partial_match)
            # return partial_matches
            

        # First find all partial patterns missing 1 part
        for missing_ind in range(len(cls.pattern)):
            part1 = cls.pattern[0:missing_ind]
            part2 = cls.pattern[missing_ind+1:]

            # print '    ',part1, part2

            if len(part1) > 0: # something to match for part1
                # print 'construction.py: 90:', part1, len(part1) > 0
                part1_matches = cls.pattern_match(part1, sequence)
                for p1match in part1_matches:
                    # print '      p1', p1match
                    hole_start = p1match[1]
                    if len(part2) > 0: # Something to match for part2 as well
                        rest_of_sequence = sequence[p1match[1]:]
                        part2_matches = \
                            cls.pattern_match(part2, rest_of_sequence)
                        for p2match in part2_matches:
                            p2match=(p1match[1]+p2match[0],p1match[1]+p2match[1])
                            # print '      p2', p2match
                            hole_end = p2match[0]
                            hole = cmn.Hole([cls.pattern[missing_ind]],
                                            sequence[hole_start:hole_end])
                            constituents = (sequence[p1match[0]:p1match[1]]+
                                            [hole]+
                                            sequence[p2match[0]:p2match[1]])
                            partial_match = cmn.Match(start=p1match[0],
                                                      end=p2match[1],
                                                      construction=cls,
                                                      constituents=constituents)
                            partial_matches.append(partial_match)
                    else: # Nothing to match for part2
                        # Hole could cover 1 item to full remainder of sequence
                        for hole_end in range(hole_start+1, len(sequence)+1):
                            hole = cmn.Hole([cls.pattern[missing_ind]],
                                            sequence[hole_start:hole_end])
                            constituents = (sequence[p1match[0]:p1match[1]]+
                                            [hole])
                            partial_match = cmn.Match(start=p1match[0],
                                                      end=hole_end,
                                                      construction=cls,
                                                      constituents=constituents)
                            partial_matches.append(partial_match)
            else: # Nothing to match for part1
                # print '  part2', part2
                part2_matches = \
                        cls.pattern_match(part2, sequence[1:])
                for p2match in part2_matches:
                    p2match = (p2match[0]+1,p2match[1]+1)
                    hole_end = p2match[0]
                    for hole_start in range(0,hole_end):
                        # print '         ',hole_start, hole_end
                        hole = cmn.Hole([cls.pattern[missing_ind]],
                                        sequence[hole_start:hole_end])
                        constituents = ([hole]+
                                        sequence[p2match[0]:p2match[1]])
                        partial_match = cmn.Match(start=hole_start,
                                                  end=p2match[1],
                                                  construction=cls,
                                                  constituents=constituents)
                        partial_matches.append(partial_match)

        return partial_matches

    def sempole(self):
        arguments = [self.constituents[ind] for ind in self.arg_indices]
        # print '78',self.__class__
        # print '79  ',self.function
        # if self.function.__name__ == 'ReturnUnaltered':
        #     print self.__class__, arguments[0].sempole()
        # print '80    ', arguments
        # print '81      ',self.function(*arguments)
        return self.function(*arguments).copy()

    def __repr__(self):
        return self.__class__.__name__

    def prettyprint(self):
        string = self.__repr__()+'\n'
        for constituent in self.constituents:
            string += cmn.indent(constituent.prettyprint())
        return string

    def collect_leaves(self):
        leaves = []
        for constituent in self.constituents:
            leaves.extend(constituent.collect_leaves())
        return leaves

    def print_sentence(self):
        leaves = self.collect_leaves()
        return ' '.join(leaves)

    def __hash__(self):
        return hash(self.prettyprint())

    def __eq__(self, other):
        return hash(self) == hash(other)