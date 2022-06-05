# Data preprocessing processes, including getting raw data from data/raw/, uploading to S3 bucket (optional, included for reference),
# and preprocessing XXX # TODO

import logging
import os
import os.path
import sys
from typing import Tuple
import boto3
import re

from botocore.exceptions import PartialCredentialsError, NoCredentialsError, ClientError

logger = logging.getLogger(__name__)


def parse_s3(s3path: str) -> Tuple[str, str]:  # TODO: NEED TO TEST THIS
    """Parse s3 path. Source: https://github.com/MSIA/2022-msia423/blob/main/aws-s3/s3.py """
    regex = r"s3://([\w._-]+)"

    m = re.match(regex, s3path)
    s3bucket = m.group(1)

    return s3bucket


def upload_file_to_s3(local_path: str, s3bucket: str, s3_just_path: str) -> None:
    """Upload objects to S3 bucket"""

    # Extract S3 bucket names from "s3://xxx"
    s3bucket = parse_s3(s3bucket)

    # connect to S3, raise errors if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
    try:
        s3 = boto3.resource('s3',
                            aws_access_key_id=os.environ.get(
                                "AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        logger.info("AWS S3 bucket connected.")
    except PartialCredentialsError as pe:  # QUESTION except here or after upload
        logger.error("Invalid AWS credentials")
        sys.exit(1)
    except NoCredentialsError as ce:  # check if it is this error
        logger.error("""No AWS credentials detected in env. Please provide credentials via AWS_ACCESS_KEY_ID and
        AWS_SECRET_ACCESS_KEY env variables""")
        sys.exit(1)

    bucket = s3.Bucket(s3bucket)

    # Check if the file already exists in S3. If not, upload the file.
    try:
        s3.Object(s3bucket, s3_just_path).load()
    except ClientError as e:
        logger.info("File does not exist in S3. Uploading file to S3.")

        # Upload the file
        try:
            bucket.upload_file(local_path, s3_just_path)
            logger.info('Data uploaded from %s to %s',
                        local_path, s3bucket+s3_just_path)
        except boto3.exceptions.S3UploadFailedError as se:
            logger.error(
                """Upload failed. Please check if the file exists and if the S3 path is correct. %s""", se)
            sys.exit(1)
    else:
        logger.info("File already exists in S3. Skipping upload.")


def download_file_from_s3(local_path: str, s3bucket: str, s3_just_path: str) -> None:
    """Download file from S3 bucket"""

    # Extract S3 bucket names from "s3://xxx"
    s3bucket = parse_s3(s3bucket)

    # connect to S3, raise errors if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
    try:
        s3 = boto3.resource('s3',
                            aws_access_key_id=os.environ.get(
                                "AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        logger.info("S3 bucket %s connected.", s3bucket)
    except PartialCredentialsError as pe:  # QUESTION except here or after upload
        logger.error("Invalid AWS credentials")
        raise pe
    except NoCredentialsError as ce:  # check if it is this error
        logger.error("""No AWS credentials detected in env. Please provide credentials via AWS_ACCESS_KEY_ID and
        # AWS_SECRET_ACCESS_KEY env variables""")
        raise ce
    else:
        bucket = s3.Bucket(s3bucket)

    try:
        bucket.download_file(s3_just_path, local_path)
    except NoCredentialsError as ne:
        logger.error(
            'Please provide AWS credentials via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables.')
        raise ne
    else:
        logger.info('Data successfully downloaded from %s to %s',
                    s3bucket+s3_just_path, local_path)
