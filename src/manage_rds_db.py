import os

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TEXT, Column, Integer, String
from sqlalchemy.orm import sessionmaker

# logger = logging.getLogger(__name__) # TODO add logger
Base = declarative_base()


class Posts(Base):
    """Data model to store a collection of posts from people and their MBTI type"""

    __tablename__ = 'posts'

    person_id = Column(Integer, primary_key=True)
    type = Column(String(4), unique=False, nullable=False)
    posts = Column(TEXT, unique=False, nullable=False)


def create_db(engine_string):
    """Create database in RDS with feature tables.
    Args:
        engine_string (str): engine string for database's creation.
    Returns:
        None
    """

    # Connect to RDS
    engine = sqlalchemy.create_engine(engine_string)

    try:
        # Create schema
        Posts.metadata.create_all(engine)
        # TODO logger.info("Database created.")
    except sqlalchemy.exc.OperationalError as oe:
        # TODO logger error, VPN?, permission?
        raise oe
