import feature as feat
import domain

########## Feature extraction functions

# For group constraints
def __group_cardinality(referent, **kwargs):
    assert(isinstance(referent, tuple))
    return len(referent)

def __referent_known(referent, **kwargs):
    return True


# For properties
# 1-part features
def __referent_rep(referent, **kwargs):
    return referent.representation.__class__

def __relatum_rep(relatum, **kwargs):
    return relatum.representation.__class__

def __referent_color(referent, **kwargs):
    return referent.color

def __referent_class(referent, **kwargs):
    return referent.object_class

def __referent_height(referent, **kwargs):
    raise NotImplementedError

def __referent_width(referent, **kwargs):
    raise NotImplementedError

def __referent_length(referent, **kwargs):
    raise NotImplementedError

def __referent_volume(referent, **kwargs):
    raise NotImplementedError

# 2-part features (for relations)

def __not_equal(referent, relatum, **kwargs):
    return referent != relatum

def __part_of(referent, relatum, **kwargs):
    return referent.get_parent_landmark() == relatum

def __contains(referent, relatum, **kwargs):
    return relatum.contains(referent)

def __distance_between(referent, relatum, **kwargs):
    distance = referent.distance_to(relatum.representation)
    # print referent, relatum, distance
    return distance

def __angle_between(referent, relatum, viewpoint, **kwargs):
    angle = relatum.angle_between(viewpoint, referent)
    # print 'gen2_features: 56', referent, relatum, angle
    return angle




group_cardinality = feat.Feature(measure_func=__group_cardinality,
                                 domain=domain.Domain('group_cardinality'))
referent_known = feat.Feature(measure_func=__referent_known,
                              domain=domain.Domain('referent_known'))
referent_rep = feat.Feature(measure_func=__referent_rep,
                            domain=domain.Domain('referent_rep'))
relatum_rep = feat.Feature(measure_func=__relatum_rep,
                           domain=domain.Domain('relatum_rep'))
referent_color = feat.Feature(measure_func=__referent_color,
                              domain=domain.Domain('referent_color'))
referent_class = feat.Feature(measure_func=__referent_class,
                              domain=domain.Domain('referent_class'))
referent_height = feat.Feature(measure_func=__referent_height,
                               domain=domain.Domain('referent_height'))
referent_width = feat.Feature(measure_func=__referent_width,
                              domain=domain.Domain('referent_width'))
referent_length = feat.Feature(measure_func=__referent_length,
                               domain=domain.Domain('referent_length'))
referent_volume = feat.Feature(measure_func=__referent_volume,
                               domain=domain.Domain('referent_volume'))
part_of = feat.Feature(measure_func=__part_of,
                       domain=domain.Domain('part_of'))
contains = feat.Feature(measure_func=__contains,
                        domain=domain.Domain('contains'))
distance_between = feat.Feature(measure_func=__distance_between,
                                domain=domain.Domain('distance_between'))
angle_between = feat.Feature(measure_func=__angle_between,
                             domain=domain.CircularDomain('angle_between',
                                                          -180, 180))