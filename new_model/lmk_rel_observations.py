#!/usr/bin/env python

from __future__ import division
# import os
import sys
sys.path.insert(1,"..")

from sqlalchemy import create_engine
from sqlalchemy import (Boolean, Column, Integer, Float, ForeignKey, Sequence, 
						String)

from automain import automain 
from argparse import ArgumentParser

from model import Base, StandardMixin
from lmk_rel_feature_model import (LinguisticRelationFeatureMixin,
								   SemanticRelationFeatureMixin,
								   LinguisticLandmarkFeatureMixin,
								   SemanticLandmarkFeatureMixin)

from utils import logger

# Base = declarative_base()
engine = None
Session = None

class Utterance(Base, StandardMixin):
	id = Column(Integer, Sequence('utterance_id_seq'), primary_key=True)
	text = Column(String(75))


class RelationObservation(Base,
						  LinguisticRelationFeatureMixin,
						  SemanticRelationFeatureMixin,
						  StandardMixin):

	id = Column(Integer, Sequence('observation_id_seq'), primary_key=True)
	# scene_id = Column(Integer, Sequence('scene_id_seq'))
	utterance_id = Column(Integer, ForeignKey('utterances.id'))

	landmark_prior = Column(Float, Sequence('landmark_prior'))
	# true_landmark = Column(String)
	true_relation_applicability = Column(Float, 
									Sequence('true_relation_applicability'))
	true_relation_score = Column(Float, Sequence('true_relation_score'))

	positive = Column(Boolean)

	@classmethod
	def make_observations(cls, session, speaker, landmark, trajector, 
						  utterance, landmark_prior, positive,
						  true_landmark, true_relation_applicability,
						  true_relation_score):

		linguistic_features = \
			cls.extract_linguistic_features(session, utterance.text)
		semantic_features = \
			cls.extract_semantic_features(speaker, landmark, trajector)

		features = dict(linguistic_features.items()+semantic_features.items())
		obsv = RelationObservation(**features)
		obsv.utterance_id = utterance.id
		obsv.landmark_prior = landmark_prior
		obsv.positive = positive
		# obsv.true_landmark = true_landmark
		obsv.true_relation_applicability = true_relation_applicability
		obsv.true_relation_score = true_relation_score
		session.add(obsv)
		session.commit()

class LandmarkObservation(Base,
						  LinguisticLandmarkFeatureMixin,
						  SemanticLandmarkFeatureMixin,
						  StandardMixin):

	id = Column(Integer, Sequence('observation_id_seq'), primary_key=True)
	# scene_id = Column(Integer, Sequence('scene_id_seq'))
	# utterance_id = Column(Integer, Sequence('utterance_id_seq'))

	landmark_prior = Column(Float, Sequence('landmark_prior'))
	# positive = Column(Boolean)

	@classmethod
	def make_observations(cls, session, speaker, landmark, 
						  landmark_prior, utterance):

		linguistic_features = \
			cls.extract_linguistic_features(session, utterance)
		semantic_features = \
			cls.extract_semantic_features(speaker, landmark)

		for lfeatures in linguistic_features:
			lfeatures.update(semantic_features)
			obsv = LandmarkObservation(**lfeatures)
			obsv.landmark_prior = landmark_prior
			session.add(obsv)
		session.commit()

@automain
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--db_url', required=True, type=str)
    args = parser.parse_args()

    engine = create_engine(args.db_url, echo=True)
    Base.metadata.create_all(engine)