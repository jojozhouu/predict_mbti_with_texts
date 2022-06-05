import argparse
import logging
import logging.config
import os.path
import random

import yaml

from src.load_save_data_s3 import upload_file_to_s3, download_file_from_s3
from src.manage_rds_db import create_db, PostManager, delete_db
from src.modeling.clean import clean_wrapper
from src.modeling.evaluate import evaluate_wrapper
from src.modeling.predict import predict_wrapper
from src.modeling.train import train_wrapper

logging.config.fileConfig('config/logging/local.conf')
logger = logging.getLogger()

# read config file instead of hardcoding
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')


if __name__ == '__main__':

    # Add parsers for both creating a database and adding songs to it
    parser = argparse.ArgumentParser(
        description="Build, train, test, and use a model to predict MBTI types based on texts")
    subparsers = parser.add_subparsers(dest="subparser_name")

    # Define subparsers and arguments for interacting with RDS
    sp_rds = subparsers.add_parser("manage_rds",
                                   description="Interact wwith RDS database, including creating databases \
                                       and ingesting data")
    sp_rds.add_argument("action",
                        choices=["create_db", "ingest_data", "delete_db"],
                        help="Action to perform on RDS database.")
    sp_rds.add_argument("--engine_string",
                        default=SQLALCHEMY_DATABASE_URI,
                        help="SQLAlchemy engine string for database connection.")
    sp_rds.add_argument("--truncate_existing_db",
                        action="store_true",
                        help="Truncate existing database before ingesting data.")
    sp_rds.add_argument("--data_file_path",
                        default="data/raw/raw_forum.csv",
                        help="Path to data file to ingest.")

    # Define subparsers and arguments for interacting with S3
    sp_s3 = subparsers.add_parser("manage_s3",
                                  description="Interact with S3 bucket, including uploading and downloading data")
    sp_s3.add_argument("action",
                       choices=["upload_data", "download_data"],
                       help="Action to perform on S3 bucket.")
    sp_s3.add_argument("--data_file_path",
                       default="data/raw/raw_forum.csv",
                       help="Path to data file to upload to S3 bucket.")
    sp_s3.add_argument("--s3_path",
                       default="s3://2022-msia423-zhou-jojo/raw/raw_forum.csv",
                       help="Path to S3 bucket directory to upload data to.")

    # Define subparsers and arguments for interacting with model
    # including cleaning data, training, predicting, and evaluating the model
    sp_model = subparsers.add_parser("model",
                                     description="Interact with model, including cleaning data, training, \
        predicting, and evaluating the model")
    sp_model.add_argument("action",
                          choices=["clean", "train",
                                   "predict", "evaluate", "all"],
                          help="Which action to run? All means running the entire pipeline.")

    # For all actions, define config file first
    sp_model.add_argument("--config_path",
                          default="config/config.yaml",
                          help="Path to the config yaml file.")

    # Define arguments if action == clean
    sp_model.add_argument("--raw_data",
                          default="data/raw/raw_forum.csv",
                          help="Path to raw data file.")
    sp_model.add_argument("--is_new_data", action="store_true",
                          help="Whether the given raw data csv is to cleaned for prediction "
                          "or, if false, for model training. Defaults to False.")
    sp_model.add_argument("-clean_data_output_dir",
                          default="data/clean",
                          help="Directory to store the cleaned data for later steps.")

    # Define arguments if action == train
    sp_model.add_argument("--clean_data_path",
                          default="data/clean/clean_posts.csv",
                          help="Path to cleaned data file.")
    sp_model.add_argument("--model_output_dir",
                          default="output/model",
                          help="Directory to store the trained model object.")
    sp_model.add_argument("--split_output_dir",
                          default="output/train_test_split",
                          help="Directory to store the train and test split.")
    sp_model.add_argument("--vectorizer_output_dir",
                          default="output/vectorizer",
                          help="Directory to store the vectorizer object.")

    # Define arguments if action == predict
    sp_model.add_argument("--model_folder_path",
                          default="output/model",
                          help="Directory where the trained model objects are stored.")
    sp_model.add_argument("--vectorizer_path",
                          default="output/vectorizer/tfidf_vectorizer.pkl",
                          help="Path to vectorizer object.")
    sp_model.add_argument("--new_text_path",
                          default="output/train_test_split/test.csv",
                          help="Either path to new text to be predicted or a string of new text.")
    sp_model.add_argument("--y_pred_output_dir",
                          default="output/predictions",
                          help="Directory to store the predictions.")
    sp_model.add_argument("--is_string",
                          action="store_true",
                          help="Whether the given `new_text_path` is a string of new text.")

    # Define arguments if action == evaluate
    sp_model.add_argument("--metrics",
                          choices=["confusion_matrix",
                                   "accuracy", "classification_report"],
                          nargs="+",
                          default=["confusion_matrix",
                                   "accuracy", "classification_report"],
                          help="Which metrics to compute?")
    sp_model.add_argument("--metrics_output_dir",
                          default="output/metrics",
                          help="Directory to store the metrics.")
    sp_model.add_argument("--y_test_path",
                          default="output/train_test_split/test.csv",
                          help="Path to test set labels.")
    sp_model.add_argument("--y_pred_folder_path",
                          default="output/predictions",
                          help="Directory where the predictions are stored.")

    # Parse arguments
    args = parser.parse_args()
    sp_used = args.subparser_name

    # Define actions related to `manage_rds`
    if sp_used == "manage_rds":

        if args.action == "create_db":
            # create database in RDS
            create_db(args.engine_string)

        elif args.action == "ingest_data":
            # Determine whether to truncate existing database
            is_truncate = 0
            if args.truncate_existing_db:
                is_truncate = 1

            # ingest data into RDS database
            post_manager = PostManager(engine_string=args.engine_string)
            post_manager.ingest_raw_data_file(
                args.data_file_path, truncate=is_truncate)

        elif args.action == "delete_db":
            # delete database in RDS
            delete_db(args.engine_string)

    # Define actions related to `manage_s3`
    elif sp_used == "manage_s3":

        if args.action == "upload_data":
            # upload data to S3
            upload_file_to_s3(args.data_file_path, args.s3_path)

        elif args.action == "download_data":
            # download raw data from S3 bucket, if raw data does not exist locally
            if not os.path.isfile(args.data_file_path):
                download_file_from_s3(
                    args.s3_path, args.data_file_path)

    # Define actions related to `model`
    elif sp_used == "model":

        # Load configuration file
        try:
            f = open(args.config_path, "r", encoding="utf-8")
        except FileNotFoundError as e:
            logger.error("Config file not found: %s", args.config_path)
            raise e
        else:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
                logger.info("Config file loaded: %s", args.config_path)
            except yaml.YAMLError as e:
                logger.error("Error parsing config file: %s", args.config_path)
                raise e
            finally:
                f.close()

        # Set Random Seed before starting
        if "random_seed" in config:
            random.seed(config["random_seed"])
            logger.info("Set random seed to %d", config["random_seed"])

        # action = "clean", cleaning raw posts
        if args.action in ["clean", "all"]:
            config_step = config["clean"]
            clean_data = clean_wrapper(args.raw_data, args.clean_data_output_dir,
                                       args.is_new_data, save_output=True, **config_step)

        # action = "train", perform train-test split, fit a vectorizer,
        # and train a logistic regression model
        if args.action in ["train", "all"]:
            config_step = config["train"]

            logger.info(
                "Performing train-test split and training a logit model")
            train_wrapper(args.clean_data_path,
                          args.model_output_dir,
                          args.vectorizer_output_dir,
                          args.split_output_dir,
                          **config_step)
            logger.info("Train step finished")

        # action = "predict", predict the fitted model on new text
        if args.action in ["predict", "all"]:
            config_step = config["predict"]

            logger.info("Predicting the model")
            # Retrieve predictors
            config_step = config["predict"]["predict_wrapper"]
            predict_wrapper(
                args.model_folder_path, args.new_text_path, args.vectorizer_path,
                args.y_pred_output_dir, args.is_string, save_output=True, **config_step)
            logger.info("Score step finished")

        # step = "evaluate", evaluate the model
        if args.action in ["evaluate", "all"]:
            config_step = config["evaluate"]["evaluate_wrapper"]

            logger.info("Evaluating the model")
            evaluate_wrapper(args.metrics,
                             args.metrics_output_dir, args.y_test_path,
                             args.y_pred_folder_path, **config_step)
            logger.info("Evaluate step finished")
