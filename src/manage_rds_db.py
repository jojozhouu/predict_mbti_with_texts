import logging
import sys
from time import time
from typing import Optional

import flask
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import TEXT, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError


logger = logging.getLogger(__name__)
Base = declarative_base()


class PostsWithLabel(Base):
    """Data model to store a collection of posts from people and their reported MBTI type"""

    __tablename__ = "posts_for_training"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String(4), unique=False, nullable=False)
    posts = Column(TEXT, unique=False, nullable=False)

    def __repr__(self):
        return f"<posts first 10 letter {self.posts[:10]}>"


class TextsFromApp(Base):
    """Data model to store raw user input from web app and
    their predicted MBTI type from the trained model"""

    __tablename__ = "posts_predicted"

    id = Column(Integer, primary_key=True, autoincrement=True)
    predicted_type = Column(String(4), unique=False, nullable=True)
    raw_text = Column(TEXT, unique=False, nullable=False)
    cleaned_text = Column(TEXT, unique=False, nullable=True)

    def __repr__(self):
        return f"<posts first 10 letter {self.raw_text[:10]}>"


def create_db(engine_string: str) -> None:
    """Create database in RDS with `posts_for_training` and `posts_predicted` tables.

    Args:
        engine_string (`str`): engine string for database's creation.

    Returns:
        None

    Raises:
        ProgrammingError, OperationalError, SQLAlchemyError: Database creation fails, maybe because
             VPN is not connected
        AttributeError: If engine string is not valid
    """
    # Connect to database
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
            # Create tables in the database
            Base.metadata.create_all(engine)
            logger.info("Database created.")
        except OperationalError as e:
            logger.error(""" Could not create database. Pleases make sure VPN is
            connected and RDS URI is correct. %s""", e)
            sys.exit(1)


def delete_db(engine_string: str) -> None:
    """Delete database

    Args:
        engine_string (`str`): engine string for database's deletion.

    Returns:
        None

    Raises:
        OperationalError: Database deletion fails, maybe because VPN is not connected.
    """

    # Connect to database
    logger.info("Deleting database...")
    engine = sqlalchemy.create_engine(engine_string)

    # Drop all the tables in the database
    try:
        Base.metadata.drop_all(engine)
    except OperationalError as e:
        logger.error(""" Could not delete database. Pleases make sure VPN is
        connected and RDS URI is correct. %s""", e)
        sys.exit(1)
    else:
        logger.info("Database deleted")


class PostManager:
    """Creates a SQLAlchemy connection to the posts table."""

    def __init__(self, app: Optional[flask.app.Flask] = None,
                 engine_string: Optional[str] = None):
        """Initialize PostManager with either a Flask app or a SQLAlchemy engine string.

        Args:
            app (:obj:`flask.app.Flask`): Flask app object for when connecting from
                within a Flask app.
            engine_string (`str`): SQLAlchemy engine string specifying which database
                to write to.

        Raises:
            ValueError: If neither app nor engine_string is provided.

        """
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

    def truncate(self, table: str) -> None:
        """Truncates the specified table.

        Args:
            table (`str`): Name of table class to truncate.

        Returns:
            None

        Raises:
            OperationalError: Errors when committing the truncate transaction.
        """
        logger.info("Truncating posts table")
        try:
            self.session.query(table).delete()
            self.session.commit()
        except OperationalError as e:
            logger.error("""Could not truncate table.
                             Does the table to be truncated exist? %s""", e)
            self.session.rollback()
        else:
            logger.info("Truncated posts table")
        finally:
            self.close()

    def ingest_raw_data_file(self, raw_data_file: str, truncate: bool = 0) -> None:
        """Ingests data from a local csv file into the posts table.

        Args:
            raw_data_file (str): path to the csv file to ingest.
            truncate (bool): whether to truncate the table before ingesting. Defaults to 0.

        Returns:
            None

        Raises:
            FileNotFoundError: If the file to ingest could not be found.
            SQLAlchemyError: Errors when committing the ingest transaction.
            OperationalError: If could not connect to database, maybe because VPN is not connected.
            ProgrammingError: If the table to ingest could not be found.
        """
        # Truncate table if specified
        if truncate:
            self.truncate(PostsWithLabel)

        logger.info("Ingesting data")
        try:
            # Read the local file to be ingested
            file = open(raw_data_file, "r", encoding="latin-1")
        except FileNotFoundError:
            logger.error("Could not find file %s", raw_data_file)
            sys.exit(1)
        else:
            logger.info("Ingesting data from %s", raw_data_file)
            next(file)
            cnt = 0
            # Ingest the file line by line
            for line in file:
                type, post = line.split(  # pylint: disable=redefined-builtin
                    ",", 1)
                try:
                    # Create a new post object and add it to the session
                    self.session.add(
                        PostsWithLabel(type=type, posts=post))
                except SQLAlchemyError as e:
                    logger.error(
                        """An error occur when adding records to the session.
                          Transaction rolled back, %s""", e)
                    self.session.rollback()
                else:
                    cnt += 1

            logger.info("Added %d records to the session", cnt)

        finally:
            file.close()

        # Commit the session
        try:
            logger.info("Committing session...It could take 5-6 minutes...")
            start_time = time()
            self.session.commit()
        except OperationalError as e:
            logger.error("""Could not find the table. Transaction rolled back.
                              Please make sure VPN is connected. Error: %s""", e)
            self.session.rollback()
        except ProgrammingError as e:
            logger.error("""Could not find the table. Transaction rolled back.
            Please make sure the tables exist.""")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Exiting due to error: %s", e)
            self.session.rollback()
        else:
            logger.info("Ingestion from %s complete. Took %.2f seconds",
                        raw_data_file, time() - start_time)
        finally:
            self.close()

    def ingest_app_user_input(self, raw_text: str, cleaned_text: str,
                              pred_type: str, truncate: bool = 0) -> None:
        """Ingests user input raw texts, cleaned texts, and the predicted MBTI type
        from the web app into posts_predicted table

        Args:
            raw_text (`str`): raw text from the user input.
            cleaned_text (`str`): cleaned user input.
            pred_type (`str`): predicted MBTI type from the user input.
            truncate (`bool`): whether to truncate the table before ingesting. Defaults to False.

        Returns:
            None

        Raises:
            SQLAlchemyError: Errors when committing the ingest transaction.
            OperationalError: If could not connect to database, maybe because VPN is not connected.
            ProgrammingError: If the table to ingest could not be found.
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
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Exiting due to error: %s", e)
            self.session.rollback()
        else:
            logger.info("Ingesting %s complete. Took %.2f seconds",
                        raw_text[:10], time() - start_time)
        finally:
            self.close()
