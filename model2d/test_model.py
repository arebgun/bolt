#!/usr/bin/env python
from __future__ import division
# import os
import sys
sys.path.insert(1,"..")


from sqlalchemy import create_engine
from sqlalchemy import (Column, Integer, Float, String, ForeignKey, DateTime, 
    Boolean, Sequence)
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import func

from automain import automain 
from argparse import ArgumentParser


Base = declarative_base()
engine = None
Session = None

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50))
    fullname = Column(String(50))
    password = Column(String(12))

    def __init__(self, name, fullname, password):
        self.name = name
        self.fullname = fullname
        self.password = password

    def __repr__(self):
        return "<User('%s','%s', '%s')>" % (self.name, self.fullname, self.password)


def get_session_factory(db_url, echo=True):
    engine = create_engine(db_url, echo=echo)
    session_factory = sessionmaker(bind=engine)
    return session_factory

@automain
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--db_url', required=True, type=str)
    args = parser.parse_args()

    engine = create_engine(args.db_url, echo=True)
    Base.metadata.create_all(engine)