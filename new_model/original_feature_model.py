#!/usr/bin/env python

from __future__ import division
# import os
import sys
sys.path.insert(1,"..")

from sqlalchemy import (Column,String,Float,Boolean,Sequence,Integer)
from model import Base, StandardMixin, get_session_factory
from original_features import featureDict
from parse import get_modparse
from nltk import ParentedTree
from utils import logger
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation


class SemanticFeatureMixin(object):
	landmark_type         = Column(String(15))
	landmark_color        = Column(String(15))
	parent_landmark_type  = Column(String(15))
	parent_landmark_color = Column(String(15))
	
	trajector_distance    = Column(Float, Sequence('trajector_distance'))
	trajector_contained   = Column(Boolean)
	trajector_in_front_of = Column(Boolean)
	trajector_behind      = Column(Boolean)
	trajector_left_of     = Column(Boolean)
	trajector_right_of    = Column(Boolean)
	
	sublmk_distance       = Column(Float, Sequence('sublmk_distance'))
	sublmk_contained      = Column(Boolean)
	sublmk_in_front_of    = Column(Boolean)
	sublmk_behind         = Column(Boolean)
	sublmk_left_of        = Column(Boolean)
	sublmk_right_of       = Column(Boolean)

	# semantic_feature_columns = ['landmark_type',
	# 							'landmark_color',
	# 							'parent_landmark_type',
	# 							'parent_landmark_color',

	# 							'trajector_distance',
	# 							'trajector_contained',
	# 							'trajector_in_front_of',
	# 							'trajector_behind',
	# 							'trajector_left_of',
	# 							'trajector_right_of',

	# 							'sublmk_distance',
	# 							'sublmk_contained',
	# 							'sublmk_in_front_of',
	# 							'sublmk_behind',
	# 							'sublmk_left_of',
	# 							'sublmk_right_of']

	@staticmethod
	def extract_semantic_features(speaker,landmark,trajector):
		features = dict()
		features['landmark_type'] = landmark.object_class
		features['landmark_color'] = landmark.color

		perspective = speaker.get_head_on_viewpoint(landmark)

		features['trajector_distance'] = \
			featureDict['distance'].measure(perspective,landmark,trajector)
		features['trajector_contained'] = \
			featureDict['contained'].measure(perspective,landmark,trajector)
		features['trajector_in_front_of'] = \
			featureDict['in_front_of'].measure(perspective,landmark,trajector)
		features['trajector_behind'] = \
			featureDict['behind'].measure(perspective,landmark,trajector)
		features['trajector_left_of'] = \
			featureDict['left_of'].measure(perspective,landmark,trajector)
		features['trajector_right_of'] = \
			featureDict['right_of'].measure(perspective,landmark,trajector)

		if landmark.parent:
			parent = landmark.parent.parent_landmark
			features['parent_landmark_type'] = parent.object_class
			features['parent_landmark_color'] = parent.color

			middle = Landmark('middle', 
							PointRepresentation(parent.representation.middle), 
							None)
			perspective = speaker.get_head_on_viewpoint(parent)

			features['sublmk_distance'] = \
				featureDict['distance'].measure(perspective,middle,landmark)
			features['sublmk_contained'] = \
				featureDict['contained'].measure(perspective,middle,landmark)
			features['sublmk_in_front_of'] = \
				featureDict['in_front_of'].measure(perspective,middle,landmark)
			features['sublmk_behind'] = \
				featureDict['behind'].measure(perspective,middle,landmark)
			features['sublmk_left_of'] = \
				featureDict['left_of'].measure(perspective,middle,landmark)
			features['sublmk_right_of'] = \
				featureDict['right_of'].measure(perspective,middle,landmark)

		return features


class LinguisticFeatureMixin(object):
	production_child = Column(String(15))
	production_parent = Column(String(15))
	production_grandparent = Column(String(15))

	linguistic_feature_columns = ['production_child',
								  'production_parent',
								  'production_grandparent']

	@classmethod
	def extract_linguistic_features(cls,session, utterance):
		parse, modparse = get_modparse(session, utterance)
		tree = ParentedTree(modparse)

		return cls.get_productions(tree)
		
	@classmethod
	def get_productions(cls,tree):
		# logger(tree)
		# logger(isinstance(tree,ParentedTree))


		if tree == tree.root: productions = [{'production_child':tree.node, 
											  'production_parent':None, 
											  'production_grandparent':None}]
		else: 
			productions = []

		grandparent = tree._parent.node if tree._parent else None
		# logger(grandparent)
		parent = tree.node
		# logger(parent)

		if isinstance(tree[0], ParentedTree):
			productions.extend(
				[{'production_child':child.node,
				  'production_parent':parent,
				  'production_grandparent':grandparent}  for child in tree])
			for t in tree:
				productions.extend( cls.get_productions(t) )
		else: 
			productions.extend(
				[{'production_child':child,
				  'production_parent':parent,
				  'production_grandparent':grandparent} for child in tree])

		return productions

