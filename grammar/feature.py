import operator

class Feature(object):
    
    def __init__(self, measure_func, domain=None):
        self.observe = measure_func
        self.domain = domain

    def __validate_other(self, other):
        assert(isinstance(other,Feature))

    def __make_measure_func(self, operator, other_feature):
        def new_measure_func(featargs):
            return operator(self.observe(featargs),
                    other_feature.observe(featargs))
        return new_measure_func

    def __make_combo_feature(self, operator, other_feature):
        self.__validate_other(other_feature)
        return Feature(operator(self.domain, other_feature.domain), 
            self.__make_measure_func(operator, other_feature))

    def __add__(self, other):
        return self.__make_combo_feature(operator.__add__, other)

    def __sub__(self, other):
        return self.__make_combo_feature(operator.__sub__, other)

    def __mul__(self, other):
        return self.__make_combo_feature(operator.__mul__, other)

    def __div__(self, other):
        return self.__make_combo_feature(operator.__div__, other)

    __truediv__ = __div__

    # def __not__(self):
    #     def not_feature_func(referent, relatum):
    #         return 1-self.observe(referent=referent, relatum=relatum)
    #     return Feature(self.domain, not_feature_func)


class FeatureArgs(object):

    def __init__(self, scene=None, speaker=None, referent=None, relatum=None):
        self.scene=scene
        self.speaker=speaker
        self.referent=referent
        self.relatum=relatum
