import logging
import logging.config
import os.path

from src.load_save_data_s3 import upload_file_to_s3, download_file_from_s3
from src.manage_rds_db import create_db, PostManager

logging.config.fileConfig('config/logging/local.conf')
logger = logging.getLogger()

if __name__ == '__main__':

    # Upload raw data files to given S3 bucket
    # upload_file_to_s3("data/raw/raw_forum.csv",
    #                  "s3://2022-msia423-zhou-jojo/raw/raw_forum.csv")

    # download raw data from S3 bucket TODO, incorporate into argparser
    # if not os.path.isfile("data/raw/raw_forum.csv"):
    #     download_file_from_s3(
    #         "s3://2022-msia423-zhou-jojo/raw/raw_forum.csv", "data/raw/raw_forum.csv")

    # create database in RDS TODO, incorporate into argparser
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
    create_db(SQLALCHEMY_DATABASE_URI)

    # ingest data into RDS database TODO, incorporate into argparser
    post_manager = PostManager(engine_string=SQLALCHEMY_DATABASE_URI)
    post_manager.ingest_data_file("data/raw/raw_forum.csv", truncate=0)
