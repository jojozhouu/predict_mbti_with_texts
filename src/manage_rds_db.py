import logging
import sys
from time import time
from typing import Optional

import flask
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TEXT, Column, Integer, String, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError, IntegrityError


logger = logging.getLogger(__name__)
Base = declarative_base()


class PostsWithLabel(Base):
    """Data model to store a collection of posts from people and their known MBTI type"""

    __tablename__ = 'posts_for_training'

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(4), unique=False, nullable=False)
    posts = Column(TEXT, unique=False, nullable=False)

    def __repr__(self):
        return f"<posts first 10 letter {self.posts[:10]}>"


class TextsFromApp(Base):
    """Data model to store raw user input from web app and
    their predicted MBTI type from the trained model"""

    __tablename__ = 'posts_predicted'

    id = Column(Integer, primary_key=True, autoincrement=True)
    predicted_type = Column(String(4), unique=False, nullable=True)
    raw_text = Column(TEXT, unique=False, nullable=False)
    cleaned_text = Column(TEXT, unique=False, nullable=True)

    def __repr__(self):
        return f"<posts first 10 letter {self.raw_text[:10]}>"


def create_db(engine_string: str) -> None:
    """Create database in RDS with feature tables.
    Args:
        engine_string (`str`): engine string for database's creation.
    Returns:
        None
    """
    # Connect to RDS
    try:
        logger.info("Connecting to database...")
        engine = sqlalchemy.create_engine(engine_string)
    except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
        logger.error(""" Could not create database. Please make sure VPN is
        connected and RDS URI is correct. %s""", e)
        sys.exit(1)
    except AttributeError as e:
        logger.error("""Could not create database. Please make sure SQLALCHEMY_DATABASE_URI is
        passed as an environment variable. %s""", e)
        sys.exit(1)
    else:
        try:
            # Create schema
            Base.metadata.create_all(engine)
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
    logger.info("Deleting database...")
    engine = sqlalchemy.create_engine(engine_string)

    try:
        Base.metadata.drop_all(engine)
    except OperationalError as e:
        logger.error(""" Could not delete database. Pleases make sure VPN is
        connected and RDS URI is correct. %s""", e)
        sys.exit(1)
    else:
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

    def truncate(self, table) -> None:
        """Truncates the specified table.
        Returns:
            None
        """
        logger.info("Truncating posts table")
        try:
            self.session.query(table).delete()
            self.session.commit()
        except SQLAlchemyError as e:
            logger.error("""An error occur when truncating the table.
                             Transaction rolled back, %s""", e)
            self.session.rollback()
        except OperationalError as e:
            logger.error("""Could not truncate table.
                             Does the table to be truncated exist? %s""", e)
            self.session.rollback()
        else:
            logger.info("Truncated posts table")
        finally:
            self.close()

    def ingest_raw_data_file(self, file: str, truncate: int = 0) -> None:
        """Ingests data from a local csv file into the posts table.
        Args:
            file (str): path to the csv file to ingest.
        Returns:
            None
        """
        # Truncate table if specified
        if truncate:
            self.truncate(PostsWithLabel)

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
                type, post = line.split(",", 1)
                try:
                    self.session.add(
                        PostsWithLabel(type=type, posts=post))
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

    def ingest_app_user_input(self, raw_text: str, cleaned_text: str, pred_type: str, truncate: int = 0) -> None:
        """Ingests user input string from the web app into posts_predicted table
        Args:
            file (str): path to the csv file to ingest.
        Returns:
            None
        """
        # Truncate table if specified
        if truncate:
            self.truncate(TextsFromApp)

        logger.info(
            "Ingesting raw input data with temporary predicted label as None")
        try:
            self.session.add(
                TextsFromApp(predicted_type=pred_type, raw_text=raw_text, cleaned_text=cleaned_text))
            logger.debug(
                "Added record (type: %s, raw_text: %s ...) to the session", type, raw_text[:10])
        except SQLAlchemyError as e:
            logger.error(
                """An error occur when adding records to the session.
                    Transaction rolled back, %s""", e)
            self.session.rollback()

        # Commit the session
        try:
            logger.info("Committing session...")
            start_time = time()
            self.session.commit()
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
            logger.info("Ingesting %s complete. Took %.2f seconds",
                        raw_text[:10], time() - start_time)
        finally:
            self.close()
