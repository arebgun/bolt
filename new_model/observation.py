#!/usr/bin/env python

from __future__ import division
# import os
import sys
sys.path.insert(1,"..")

from sqlalchemy import create_engine
from sqlalchemy import (Boolean, Column, Integer, Float, Sequence)

from automain import automain 
from argparse import ArgumentParser

from model import Base, StandardMixin
from original_feature_model import SemanticFeatureMixin, LinguisticFeatureMixin

from utils import logger

# Base = declarative_base()
engine = None
Session = None

class Observation(Base,
				  LinguisticFeatureMixin,
				  SemanticFeatureMixin,
				  StandardMixin):

	id = Column(Integer, Sequence('observation_id_seq'), primary_key=True)
	# scene_id = Column(Integer, Sequence('scene_id_seq'))
	# utterance_id = Column(Integer, Sequence('utterance_id_seq'))

	landmark_prior = Column(Float, Sequence('landmark_prior'))
	positive = Column(Boolean)

	@classmethod
	def make_observations(cls, session, speaker, landmark, 
						  trajector, utterance, positive):

		linguistic_features = \
			cls.extract_linguistic_features(session, utterance)
		semantic_features = \
			cls.extract_semantic_features(speaker, landmark, trajector)

		for lfeatures in linguistic_features:
			lfeatures.update(semantic_features)
			obsv = Observation(**lfeatures)
			obsv.landmark_prior = 1.0
			obsv.positive = positive
			session.add(obsv)
		session.commit()

@automain
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--db_url', required=True, type=str)
    args = parser.parse_args()

    engine = create_engine(args.db_url, echo=True)
    Base.metadata.create_all(engine)