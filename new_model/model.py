from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import Column, DateTime
from sqlalchemy import func, create_engine
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

class StandardMixin(object):
	@declared_attr
	def __tablename__(cls):
		return cls.__name__.lower() + 's'

	created_at = Column(DateTime, default=func.now())

def get_session_factory(db_url, echo=True):
    engine = create_engine(db_url, echo=echo)
    session_factory = sessionmaker(bind=engine)
    return session_factory