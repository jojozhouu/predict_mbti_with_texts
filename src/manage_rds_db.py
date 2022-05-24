import logging
import os
import sys
from time import time
from typing import Optional

import flask
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TEXT, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError, IntegrityError
from sqlalchemy_utils.functions import database_exists


logger = logging.getLogger(__name__)
Base = declarative_base()


class Posts(Base):
    """Data model to store a collection of posts from people and their MBTI type"""

    __tablename__ = 'posts'

    person_id = Column(Integer, primary_key=True)
    type = Column(String(4), unique=False, nullable=False)
    posts = Column(TEXT, unique=False, nullable=False)

    def __repr__(self):
        return f"<posts first 10 letter {self.posts[:10]}>"


def create_db(engine_string: str) -> None:
    """Create database in RDS with feature tables.
    Args:
        engine_string (`str`): engine string for database's creation.
    Returns:
        None
    """
    # Connect to RDS

    engine = sqlalchemy.create_engine(engine_string)
    try:
        # Create schema
        Posts.metadata.create_all(engine)
        logger.info("Database created.")
    except OperationalError as e:
        logger.error(""" Could not create database. Pleases make sure VPN is 
        connected and RDS URI is correct. %s""", e)
        sys.exit(1)


def delete_db(engine_string: str) -> None:
    """Delete database in RDS .
    Args:
        engine_string (`str`): engine string for database's deletion.
    Returns:
        None
    """
    # The Base.metadata object collects and manages Table operations
    engine = sqlalchemy.create_engine(engine_string)

    try:
        Posts.metadata.drop_all(engine)
    except OperationalError as e:
        logger.error(""" Could not delete database. Pleases make sure VPN is
        connected and RDS URI is correct. %s""", e)
        sys.exit(1)
    else:
        pass
        logger.info("Database deleted")


class PostManager:
    """Creates a SQLAlchemy connection to the posts table.
    Args:
        app (:obj:`flask.app.Flask`): Flask app object for when connecting from
            within a Flask app. Optional.
        engine_string (`str`): SQLAlchemy engine string specifying which database
            to write to. Follows the format
    """

    def __init__(self, app: Optional[flask.app.Flask] = None,
                 engine_string: Optional[str] = None):
        if app:
            self.database = SQLAlchemy(app)
            self.session = self.database.session
        elif engine_string:
            engine = sqlalchemy.create_engine(engine_string)
            session_maker = sessionmaker(bind=engine)
            self.session = session_maker()
        else:
            raise ValueError(
                "Need either an engine string or a Flask app to initialize")

    def close(self) -> None:
        """Closes SQLAlchemy session."""
        self.session.close()

    def truncate(self) -> None:
        """Truncates the posts table.
        Returns:
            None
        """
        logger.info("Truncating posts table")
        try:
            self.session.query(Posts).delete()
            self.session.commit()
        except SQLAlchemyError as e:
            logger.error("""An error occur when truncating the table.
                             Transaction rolled back, %s""", e)
            self.session.rollback()
        else:
            logger.info("Truncated posts table")
        finally:
            self.close()

    def ingest_data_file(self, file: str, truncate: int = 0) -> None:
        """Ingests data from a local csv file into the posts table.
        Args:
            file (str): path to the csv file to ingest.
        Returns:
            None
        """
        # Truncate table if specified
        if truncate:
            self.truncate()

        logger.info("Ingesting data")
        try:
            f = open(file, 'r', encoding='latin-1')
        except FileNotFoundError as fe:
            logger.error("Could not find file {}".format(file))
            sys.exit(1)
        else:
            logger.info("Ingesting data from {}".format(file))
            next(f)
            cnt = 0
            for line in f:
                id, type, post = line.split(",", 2)
                try:
                    self.session.add(
                        Posts(person_id=id, type=type, posts=post))
                except SQLAlchemyError as e:
                    logger.error(
                        """An error occur when adding records to the session.
                          Transaction rolled back, %s""", e)
                    self.session.rollback()
                else:
                    cnt += 1
                    # logger.debug(
                    #     "Added record (type: %s, post: %s ...) to the session", type, post[:10])

            logger.info("Added %d records to the session", cnt)

        finally:
            f.close()

        # Commit the session
        try:
            start_time = time()
            self.session.commit()
        except IntegrityError as e:
            logger.error("""The database already contains the records you are trying to insert.
            Transaction rolled back. Please truncate the table before attempting again. """)
            self.session.rollback()
        except OperationalError as e:
            logger.error("""Could not find the table. Transaction rolled back.
                              Please make sure VPN is connected. Error: %s""", e)
            self.session.rollback()
        except ProgrammingError as e:
            logger.error("""Could not find the table. Transaction rolled back. 
            Please make sure the table exists.""")
        except Exception as e:
            logger.error("Exiting due to error: %s", e)
            self.session.rollback()
        else:
            logger.info("Ingestion from %s complete. Took %.2f seconds",
                        file, time() - start_time)
        finally:
            self.close()
