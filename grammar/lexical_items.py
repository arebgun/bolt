import construction as struct
import constraint as const
import sempoles as sp


class Space(struct.LexicalItem):
    pass
_ = Space(regex=' ', sempole=None)


class Article(struct.LexicalItem):
    pass

def known(context, entity):
    return context.foreknowledge(entity)

# a_constraints = sp.ConstraintCollection([
#     CardinalityConstraint((1,1)),
#     # PreviousKnowledgeConstraint(False)
#     ])
# a    = Article(regex='a', sempole=a_constraints)

the_constraints = const.ConstraintCollection([sp.known_property])
the  = Article(regex='the', sempole=the_constraints)


class Noun(struct.LexicalItem):
    pass


objct_constraints = const.ConstraintCollection([sp.object_property])
objct    = Noun(regex='object', sempole=objct_constraints)

# cube     = Noun(regex='cube', sempole=None) #TODO

block_constraints = const.ConstraintCollection([sp.box_property])
block    = Noun(regex='block', sempole=block_constraints)
box      = Noun(regex='box', sempole=block_constraints)

sphere_constraints = const.ConstraintCollection([sp.sphere_property])
sphere   = Noun(regex='sphere', sempole=sphere_constraints)
ball     = Noun(regex='ball', sempole=sphere_constraints)

cone_constraints = const.ConstraintCollection([sp.cone_property])
cone     = Noun(regex='cone', sempole=cone_constraints)

cylinder_constraints = const.ConstraintCollection([sp.cylinder_property])
cylinder = Noun(regex='cylinder', sempole=cylinder_constraints)

table_constraints = const.ConstraintCollection([sp.table_property])
table = Noun(regex='table', sempole=table_constraints)


corner_constraints = const.ConstraintCollection([sp.corner_property])
corner   = Noun(regex='corner', sempole=corner_constraints)
corner.lmk = True

edge_constraints = const.ConstraintCollection([sp.edge_property])
edge     = Noun(regex='edge', sempole=edge_constraints)
edge.lmk = True

end_constraints = const.ConstraintCollection([sp.end_property])
end      = Noun(regex='end', sempole=end_constraints)
end.lmk = True

half_constraints = const.ConstraintCollection([sp.half_property])
half     = Noun(regex='half', sempole=half_constraints)
half.lmk = True

middle_constraints = const.ConstraintCollection([sp.middle_property])
middle   = Noun(regex='middle', sempole=middle_constraints)
middle.lmk = True

side_constraints = const.ConstraintCollection([sp.side_property])
side     = Noun(regex='side', sempole=side_constraints)
side.lmk = True


# class Plural(struct.LexicalItem):
#     pass

# _s_constraints = const.ConstraintCollection([
#     ])
# _s = Plural(regex='(?<!:\s)s(?=[\s",;.!?])', sempole=None)


class Adjective(struct.LexicalItem):
    pass

class GradableAdjective(Adjective, struct.Gradable):
    pass

big       = GradableAdjective(regex='big', sempole=None)
large     = GradableAdjective(regex='large', sempole=None)
small     = GradableAdjective(regex='small', sempole=None)
little    = GradableAdjective(regex='little', sempole=None)
#short
#tall
#wide
#skinny

red_constraints = const.ConstraintCollection([sp.red_property])
red       = Adjective(regex='red', sempole=red_constraints)

orange_constraints = const.ConstraintCollection([sp.orange_property])
orange    = Adjective(regex='orange', sempole=orange_constraints)

yellow_constraints = const.ConstraintCollection([sp.yellow_property])
yellow    = Adjective(regex='yellow', sempole=yellow_constraints)

green_constraints = const.ConstraintCollection([sp.green_property])
green     = Adjective(regex='green', sempole=green_constraints)

blue_constraints = const.ConstraintCollection([sp.blue_property])
blue      = Adjective(regex='blue', sempole=blue_constraints)

purple_constraints = const.ConstraintCollection([sp.purple_property])
purple    = Adjective(regex='purple', sempole=purple_constraints)

pink_constraints = const.ConstraintCollection([sp.pink_property])
pink    = Adjective(regex='pink', sempole=pink_constraints)

black_constraints = const.ConstraintCollection([sp.black_property])
black     = Adjective(regex='black', sempole=black_constraints)

white_constraints = const.ConstraintCollection([sp.white_property])
white     = Adjective(regex='white', sempole=white_constraints)

gray_constraints = const.ConstraintCollection([sp.gray_property])
gray      = Adjective(regex='gray', sempole=gray_constraints)
grey      = Adjective(regex='grey', sempole=gray_constraints)


class DegreeModifier(struct.LexicalItem):
    pass

very      = DegreeModifier(regex='very', sempole=None)
somewhat  = DegreeModifier(regex='somewhat', sempole=None)
pretty    = DegreeModifier(regex='pretty', sempole=None)
extremely = DegreeModifier(regex='extremely', sempole=None)


class Direction(struct.LexicalItem):
    pass

front     = Direction(regex='front', sempole=sp.front_func)
back      = Direction(regex='back', sempole=sp.back_func)
left      = Direction(regex='left', sempole=sp.left_func)
right     = Direction(regex='right', sempole=sp.right_func)
# north     = Direction(regex='north', sempole=None)
# south     = Direction(regex='south', sempole=None)
# east      = Direction(regex='east', sempole=None)
# west      = Direction(regex='west', sempole=None)

class SourceTag(struct.LexicalItem):
    pass

frm = SourceTag(regex='from', sempole=None)


class DestinationTag(struct.LexicalItem):
    pass

to = DestinationTag(regex='to', sempole=None)


class BelongingTag(struct.LexicalItem):
    pass

of = BelongingTag(regex='of', sempole=None)


class Measure(struct.LexicalItem):
    pass

class GradableMeasure(Measure, struct.Gradable):
    pass

# class Number(struct.LexicalItem):
#     pass

# class Unit(struct.LexicalItem):
#     pass


# class ConcreteMeasure(Measure):
#     pass

class DistanceMeasure(GradableMeasure):
    pass

far = DistanceMeasure(regex='far', sempole=sp.far_func)
near = DistanceMeasure(regex='near', sempole=sp.near_func)




lexical_items_list = [
    _,
    # a,
    the,
    objct,
    # cube,
    block,
    box,
    sphere,
    ball,
    cone,
    cylinder,
    # _s,
    # big,
    # large,
    # small,
    # little,
    red,
    orange,
    yellow,
    green,
    blue,
    purple,
    pink,
    black,
    white,
    gray,
    grey,
    # very,
    # somewhat,
    # pretty,
    # extremely,
    front,
    back,
    left,
    right,
    # north,
    # south,
    # east,
    # west,
    to,
    frm,
    of,
    far,
    near,
]
