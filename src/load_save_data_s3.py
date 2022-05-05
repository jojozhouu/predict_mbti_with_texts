# Data preprocessing processes, including getting raw data from data/raw/, uploading to S3 bucket (optional, included for reference),
# and preprocessing XXX # TODO

import os
import os.path
from typing import Tuple
import boto3
import re


def parse_s3(s3path: str) -> Tuple[str, str]:  # TODO: NEED TO TEST THIS
    """Parse s3 path. Source: https://github.com/MSIA/2022-msia423/blob/main/aws-s3/s3.py """
    regex = r"s3://([\w._-]+)/([\w./_-]+)"

    m = re.match(regex, s3path)
    s3bucket = m.group(1)
    s3path = m.group(2)

    return s3bucket, s3path


def upload_file_to_s3(local_path: str, s3path: str) -> None:
    """Upload objects to S3 bucket"""

    # Extract S3 bucket names and path from input
    s3bucket, s3_just_path = parse_s3(s3path)

    # connect to S3, raise errors if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
    try:
        s3 = boto3.resource('s3',
                            aws_access_key_id=os.environ.get(
                                "AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        # TODO logger.info("AWS S3 bucket connected.")
    except botocore.exceptions.PartialCredentialsError as pe:  # QUESTION except here or after upload
        # TODO logger.error("Invalid AWS credentials")
        raise pe
    except botocore.exceptions.NoCredentialsError as ce:  # check if it is this error
        # TODO logger.error("""No AWS credentials detected in env. Please provide credentials via AWS_ACCESS_KEY_ID and
        # AWS_SECRET_ACCESS_KEY env variables""")
        raise ce

    bucket = s3.Bucket(s3bucket)

    try:
        bucket.upload_file(local_path, s3_just_path)
        # TODO, logger.info('Data uploaded from %s to %s', local_path, s3path)
    except boto3.exceptions.S3UploadFailedError as se:
        # TODO, add logging for reporting errors
        raise se


def download_file_from_s3(local_path: str, s3path: str) -> None:
    """Download file from S3 bucket"""

    # Extract S3 bucket names and path from input
    s3bucket, s3_just_path = parse_s3(s3path)

    # connect to S3, raise errors if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
    try:
        s3 = boto3.resource('s3',
                            aws_access_key_id=os.environ.get(
                                "AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        # TODO logger.info("AWS S3 bucket connected.")
    except botocore.exceptions.PartialCredentialsError as pe:  # QUESTION except here or after upload
        # TODO logger.error("Invalid AWS credentials")
        raise pe
    except botocore.exceptions.NoCredentialsError as ce:  # check if it is this error
        # TODO logger.error("""No AWS credentials detected in env. Please provide credentials via AWS_ACCESS_KEY_ID and
        # AWS_SECRET_ACCESS_KEY env variables""")
        raise ce

    bucket = s3.Bucket(s3bucket)

    # TODO logging: successfully connected

    try:
        bucket.download_file(s3_just_path, local_path)
    except botocore.exceptions.NoCredentialsError as ne:
        # logger.error(
        #    'Please provide AWS credentials via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables.')
        raise ne
    else:
        # logger.info('Data downloaded from %s to %s', s3path, local_path)
        pass
