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

# a_constraints = sp.ConstraintSet([
#     CardinalityConstraint((1,1)),
#     # PreviousKnowledgeConstraint(False)
#     ])
# a    = Article(regex='a', sempole=a_constraints)

empty_constraints = const.ConstraintSet([sp.known_property])
the  = Article(regex='the', sempole=empty_constraints)


class Noun(struct.LexicalItem):
    pass


objct_constraints = const.ConstraintSet([sp.object_property])
objct    = Noun(regex='object', sempole=objct_constraints)

# cube     = Noun(regex='cube', sempole=None) #TODO

block_constraints = const.ConstraintSet([sp.box_property])
block    = Noun(regex='block', sempole=block_constraints)
box      = Noun(regex='box', sempole=block_constraints)

sphere_constraints = const.ConstraintSet([sp.sphere_property])
sphere   = Noun(regex='sphere', sempole=sphere_constraints)
ball     = Noun(regex='ball', sempole=sphere_constraints)

cone_constraints = const.ConstraintSet([sp.cone_property])
cone     = Noun(regex='cone', sempole=cone_constraints)

cylinder_constraints = const.ConstraintSet([sp.cylinder_property])
cylinder = Noun(regex='cylinder', sempole=cylinder_constraints)

table_constraints = const.ConstraintSet([sp.table_property])
table = Noun(regex='table', sempole=table_constraints)


corner_constraints = const.ConstraintSet([sp.corner_property])
corner   = Noun(regex='corner', sempole=corner_constraints)
corner.lmk = True

edge_constraints = const.ConstraintSet([sp.edge_property])
edge     = Noun(regex='edge', sempole=edge_constraints)
edge.lmk = True

end_constraints = const.ConstraintSet([sp.end_property])
end      = Noun(regex='end', sempole=end_constraints)
end.lmk = True

half_constraints = const.ConstraintSet([sp.half_property])
half     = Noun(regex='half', sempole=half_constraints)
half.lmk = True

middle_constraints = const.ConstraintSet([sp.middle_property])
middle   = Noun(regex='middle', sempole=middle_constraints)
middle.lmk = True

side_constraints = const.ConstraintSet([sp.side_property])
side     = Noun(regex='side', sempole=side_constraints)
side.lmk = True


# class Plural(struct.LexicalItem):
#     pass

# _s_constraints = const.ConstraintSet([
#     ])
# _s = Plural(regex='(?<!:\s)s(?=[\s",;.!?])', sempole=None)


class Adjective(struct.LexicalItem):
    pass

class GradableAdjective(Adjective, struct.Gradable):
    pass

# big       = GradableAdjective(regex='big', sempole=None)
# large     = GradableAdjective(regex='large', sempole=None)
# small     = GradableAdjective(regex='small', sempole=None)
# little    = GradableAdjective(regex='little', sempole=None)
#short
#tall
#wide
#skinny

red_constraints = const.ConstraintSet([sp.red_property])
red       = Adjective(regex='red', sempole=red_constraints)

orange_constraints = const.ConstraintSet([sp.orange_property])
orange    = Adjective(regex='orange', sempole=orange_constraints)

yellow_constraints = const.ConstraintSet([sp.yellow_property])
yellow    = Adjective(regex='yellow', sempole=yellow_constraints)

green_constraints = const.ConstraintSet([sp.green_property])
green     = Adjective(regex='green', sempole=green_constraints)

blue_constraints = const.ConstraintSet([sp.blue_property])
blue      = Adjective(regex='blue', sempole=blue_constraints)

purple_constraints = const.ConstraintSet([sp.purple_property])
purple    = Adjective(regex='purple', sempole=purple_constraints)

pink_constraints = const.ConstraintSet([sp.pink_property])
pink    = Adjective(regex='pink', sempole=pink_constraints)

black_constraints = const.ConstraintSet([sp.black_property])
black     = Adjective(regex='black', sempole=black_constraints)

white_constraints = const.ConstraintSet([sp.white_property])
white     = Adjective(regex='white', sempole=white_constraints)

gray_constraints = const.ConstraintSet([sp.gray_property])
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

far = DistanceMeasure(regex='far from', sempole=sp.far_func)
near = DistanceMeasure(regex='near to', sempole=sp.near_func)


class OnRelation(struct.LexicalItem, struct.Relation):
    pass

on = OnRelation(regex='on', sempole=const.ConstraintSet([sp.on_property]))
in_ = OnRelation(regex='in', sempole=const.ConstraintSet([sp.on_property]))

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
    on,
]


#New items
a_article = Article(regex='a', sempole=empty_constraints)
an_article = Article(regex='an', sempole=empty_constraints)
'''
nouns:
top 37 227
bottom 8 72
back 19 396
rear 3 52
front 56 392
forefront 1 3
center 40 460
centre 4 74
shape 224 58
circle 151 84
prism 179 72
cube 775 291
cuboid 129 47
disc 24 18
disk 24 18
parallelepiped 7 0
'''

# import gen2_features as feats
# top_constraints = const.ConstraintSet([
#                 const.RelationConstraint(feature=feats.angle_between,
#                                          prob_func=sp.back_func)
#                 ])

# top_noun = Noun(regex='top', sempole=top_constraints)
# top_noun.lmk = True
# bottom_noun = Noun(regex='bottom', sempole=None)
# back_noun = Noun(regex='back', sempole=None)
# rear_noun = Noun(regex='rear', sempole=None)
# front_noun = Noun(regex='front', sempole=None)
# forefront_noun = Noun(regex='forefront', sempole=None)
center_noun = Noun(regex='center', sempole=middle_constraints)
center_noun.lmk = True
# centre_noun = Noun(regex='centre', sempole=None)
shape_noun = Noun(regex='shape', sempole=objct_constraints)
circle_noun = Noun(regex='circle', sempole=cylinder_constraints)
prism_noun = Noun(regex='prism', sempole=block_constraints)
cube_noun = Noun(regex='cube', sempole=block_constraints)
cuboid_noun = Noun(regex='cuboid', sempole=block_constraints)
rectangle_noun = Noun(regex='rectangle', sempole=block_constraints)
square_noun = Noun(regex='square', sempole=block_constraints)
# disc_noun = Noun(regex='disc', sempole=None)
# disk_noun = Noun(regex='disk', sempole=None)
# parallelepiped_noun = Noun(regex='parallelepiped', sempole=None)
'''
adjectives:
lower 0 25
upper 1 73
near 57 711
far 28 309
top 37 227
bottom 8 71
leftmost 1 9
rightmost 1 5
cylindrical 67 27
circular 87 35
spherical 24 22
rectangular 533 191
square 392 122
colored 34 52
coloured 139 13
brown 156 100
violet 75 34
black 11 22
-upright 21 4
-big 50 16
-large 73 19
-small 389 130
-tall 61 7
-short 77 32
-flat    10

'''
# lower_adj = Adjective(regex='lower', sempole=None)
# upper_adj = Adjective(regex='upper', sempole=None)
# near_constraints = const.ConstraintSet([
#                 const.RelationConstraint(feature=feats.angle_between,
#                                          prob_func=sp.front_func)
#                 ])
# far_constraints = const.ConstraintSet([
#                 const.RelationConstraint(feature=feats.angle_between,
#                                          prob_func=sp.back_func)
#                 ])
near_dir = Direction(regex='near', sempole=sp.front_func)
far_dir = Direction(regex='far', sempole=sp.back_func)
top_dir = Direction(regex='top', sempole=sp.back_func)
# bottom_adj = Adjective(regex='bottom', sempole=None)
# cylindrical_adj = Adjective(regex='cylindrical', sempole=None)
circular_adj = Adjective(regex='circular', sempole=cylinder_constraints)
round_adj = Adjective(regex='round', sempole=cylinder_constraints)
rectangular_adj = Adjective(regex='rectangular', sempole=block_constraints)
square_adj = Adjective(regex='square', sempole=block_constraints)
colored_adj = Adjective(regex='colored', sempole=empty_constraints)
shaped_adj = Adjective(regex='shaped', sempole=empty_constraints)
brown_adj = Adjective(regex='brown', sempole=orange_constraints)
violet_adj = Adjective(regex='violet', sempole=purple_constraints)
# black_adj = Adjective(regex='black', sempole=None)

# big_adj = Adjective(regex='big', sempole=None)
large_adj = Adjective(regex='large', sempole=empty_constraints)
small_adj = Adjective(regex='small', sempole=empty_constraints)
# tall_adj = Adjective(regex='tall', sempole=None)
short_adj = Adjective(regex='short', sempole=empty_constraints)
# wide_adj = Adjective(regex='wide', sempole=None)

'''
adverbs:
almost
slightly
right (as in directly)
pale (pink)
light (yellow)
'''
just = None
slightly = None
very = None
'''
relations:
at 537
next (to) 382
above 98
below 70
closer 48
towards 169
almost in 
across from 15
beside 64
to the rear of
in front of
away from 137
near 711
off center
adjacent to 34
anterior to 0
furthest (away) 73
closest (to) 189
nearest (to) 44
-between 196
'''
class Relation(struct.LexicalItem,struct.Relation):
    pass

import gen2_features as feats
near_constraints = const.ConstraintSet([
                const.RelationConstraint(feature=feats.distance_between,
                                         prob_func=sp.near_func)
                ])

far_constraints = const.ConstraintSet([
                const.RelationConstraint(feature=feats.distance_between,
                                         prob_func=sp.far_func)
                ])

behind_constraints = const.ConstraintSet([
                const.RelationConstraint(feature=feats.angle_between,
                                         prob_func=sp.back_func)
                ])

in_front_of_constraints = const.ConstraintSet([
                const.RelationConstraint(feature=feats.angle_between,
                                         prob_func=sp.front_func)
                ])

at_rel = Relation(regex='at', sempole=near_constraints)
next_to_rel = Relation(regex='next to', sempole=near_constraints)
near_rel = Relation(regex='near', sempole=near_constraints)
close_to_rel = Relation(regex='close to', sempole=near_constraints)
towards_rel = Relation(regex='towards', sempole=near_constraints)
away_from_rel = Relation(regex='away from', sempole=far_constraints)
in_front_of_rel = Relation(regex='in front of', sempole=in_front_of_constraints)
behind_rel = Relation(regex='behind', sempole=behind_constraints)
above_rel = Relation(regex='above', sempole=behind_constraints)

'''
miscellaneous:
a    792
an    69
and    830
but
,
;

other issues:
missing 'the'
'located', 'placed', 'situated', 'kept'
misspellings
missing commas
'''