#!/usr/bin/python
from __future__ import division
import sys
sys.path.insert(1, '..')
from semantics.representation import (PointRepresentation, LineRepresentation,
                                RectangleRepresentation, SurfaceRepresentation)
import probability_function as pfunc
import constraint as const
import gen2_features as feats


######### for LexicalItems #########

# Articles

is_true = pfunc.DiscreteProbFunc([(True, 1.0)])
known_property = const.PropertyConstraint(feature=feats.referent_known,
                                          prob_func=is_true)

# Objects

is_object = pfunc.DiscreteProbFunc([('BOX',      1.0),
                                    ('CONE',     1.0),
                                    ('CYLINDER', 1.0),
                                    ('SPHERE',   1.0)])
object_property = const.PropertyConstraint(feature=feats.referent_class,
                                           prob_func=is_object)

is_box = pfunc.DiscreteProbFunc([('BOX', 1.0)])
box_property = const.PropertyConstraint(feature=feats.referent_class,
                                        prob_func=is_box)

is_cone = pfunc.DiscreteProbFunc([('CONE', 1.0)])
cone_property = const.PropertyConstraint(feature=feats.referent_class,
                                         prob_func=is_cone)

is_cylinder = pfunc.DiscreteProbFunc([('CYLINDER', 1.0)])
cylinder_property = const.PropertyConstraint(feature=feats.referent_class,
                                             prob_func=is_cylinder)

is_sphere = pfunc.DiscreteProbFunc([('SPHERE', 1.0)])
sphere_property = const.PropertyConstraint(feature=feats.referent_class,
                                           prob_func=is_sphere)

is_table = pfunc.DiscreteProbFunc([('TABLE', 1.0)])
table_property = const.PropertyConstraint(feature=feats.referent_class,
                                          prob_func=is_table)


# Object parts

is_edge = pfunc.DiscreteProbFunc([('EDGE', 1.0)])
edge_property = const.PropertyConstraint(feature=feats.referent_class,
                                         prob_func=is_edge)

is_corner = pfunc.DiscreteProbFunc([('CORNER', 1.0)])
corner_property = const.PropertyConstraint(feature=feats.referent_class,
                                           prob_func=is_corner)

is_middle = pfunc.DiscreteProbFunc([('MIDDLE', 1.0)])
middle_property = const.PropertyConstraint(feature=feats.referent_class,
                                           prob_func=is_middle)


# Colors

is_red = pfunc.DiscreteProbFunc([('RED', 1.0)])
red_property = const.PropertyConstraint(feature=feats.referent_color,
                                        prob_func=is_red)

is_orange = pfunc.DiscreteProbFunc([('ORANGE', 1.0)])
orange_property = const.PropertyConstraint(feature=feats.referent_color,
                                           prob_func=is_orange)

is_yellow = pfunc.DiscreteProbFunc([('YELLOW', 1.0)])
yellow_property = const.PropertyConstraint(feature=feats.referent_color,
                                           prob_func=is_yellow)

is_green = pfunc.DiscreteProbFunc([('GREEN', 1.0)])
green_property = const.PropertyConstraint(feature=feats.referent_color,
                                          prob_func=is_green)

is_blue = pfunc.DiscreteProbFunc([('BLUE', 1.0)])
blue_property = const.PropertyConstraint(feature=feats.referent_color,
                                         prob_func=is_blue)

is_purple = pfunc.DiscreteProbFunc([('PURPLE', 1.0)])
purple_property = const.PropertyConstraint(feature=feats.referent_color,
                                           prob_func=is_purple)

is_pink = pfunc.DiscreteProbFunc([('PINK', 1.0)])
pink_property = const.PropertyConstraint(feature=feats.referent_color,
                                           prob_func=is_pink)

is_black = pfunc.DiscreteProbFunc([('BLACK', 1.0)])
black_property = const.PropertyConstraint(feature=feats.referent_color,
                                          prob_func=is_black)

is_white = pfunc.DiscreteProbFunc([('WHITE', 1.0)])
white_property = const.PropertyConstraint(feature=feats.referent_color,
                                          prob_func=is_white)

is_gray = pfunc.DiscreteProbFunc([('GRAY', 1.0)])
gray_property = const.PropertyConstraint(feature=feats.referent_color,
                                         prob_func=is_gray)

# Degrees

# Distance measures
far_func = pfunc.LogisticSigmoid(loc=0.55, scale=0.1, domain=None)

near_func = pfunc.LogisticSigmoid(loc=0.15, scale=-0.1, domain=None)


# Directions
front_func = pfunc.LogisticBell(loc=0, scale=30, domain=None)

back_func = pfunc.LogisticBell(loc=180, scale=30, domain=None)

left_func = pfunc.LogisticBell(loc=90, scale=30, domain=None)

right_func = pfunc.LogisticBell(loc=-90, scale=30, domain=None)

# For Semi-Constructions (Relations)

# For Containment Relations
one_true = pfunc.DiscreteProbFunc([(1.0, 1.0)])
contains_property = const.RelationConstraint(feature=feats.contains,
                                             prob_func=one_true)


# Common to Distance and Orientation Relations
zero_false = pfunc.DiscreteProbFunc([(0.0, 1.0)])
not_contains_property = const.RelationConstraint(feature=feats.contains,
                                                 prob_func=zero_false)

is_not_surface = pfunc.DiscreteProbFunc([(PointRepresentation,     1.0),
                                         (LineRepresentation,      1.0),
                                         (RectangleRepresentation, 1.0),
                                         (SurfaceRepresentation,   0.0)])
not_surface_property = const.PropertyConstraint(feature=feats.referent_rep,
                                                prob_func=is_not_surface)

# Semi-Constructions

# def PartOfRelate()

# def ContainmentRelate()

def DistanceRelate(self, distance_measure):
    return const.ConstraintCollection([
                not_surface_property,
                not_contains_property,

                const.RelationConstraint(feature=feats.distance_between,
                    prob_func=distance_measure.sempole())
           ])
    

def OrientationRelate(self, direction):
    return const.ConstraintCollection([
                not_surface_property,
                not_contains_property,
                const.RelationConstraint(feature=feats.angle_between,
                                         prob_func=direction.sempole())
           ])


########## for Constructions #########

def ReturnUnaltered(self, construction_instance):
    return construction_instance.sempole()

def DegreeModify(self, degree_modifier, gradable):
    return degree_modifier.sempole().modify(gradable.sempole())

def PropertyCombine(self, modifier, original):
    return modifier.sempole().modify(original.sempole())

def ArticleCombine(self, article, original):
    return article.sempole().modify(original.sempole())

def RelateToLandmark(self, relation, lmk_phrase):
    new_sempole = relation.sempole()
    new_sempole.relatum_constraints = lmk_phrase.sempole()
    return new_sempole

def NounPhraseRelate(self, noun_phrase, relation_landmark_phrase):
    new_sempole = noun_phrase.sempole()
    new_sempole['relation'] = relation_landmark_phrase.sempole()
    return new_sempole