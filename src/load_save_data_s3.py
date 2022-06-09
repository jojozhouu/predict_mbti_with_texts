import logging
import os
import os.path
import sys
from typing import Tuple
import boto3
import re

from botocore.exceptions import PartialCredentialsError, NoCredentialsError, ClientError

logger = logging.getLogger(__name__)

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def parse_s3(s3path: str) -> str:
    """Parse s3 path and decompose it into bucket and file path
    Source: https: // github.com/MSIA/2022-msia423/blob/main/aws-s3/s3.py 

    Args:
        s3path (str): s3 path to file

    Returns:
        S3 bucket name

    Raises:
        ValueError: if s3path does not start with "s3://"
    """

    # Raise error if s3 path does not start with "s3://"
    if not s3path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")

    # Define regex for s3 path format
    regex = r"s3://([\w._-]+)"
    m = re.match(regex, s3path)

    # Return only the bucket name
    s3bucket = m.group(1)

    return s3bucket


def upload_file_to_s3(local_path: str, s3bucket: str, s3_just_path: str) -> None:
    """Upload files to S3 bucket

    Args:
        local_path (`str`): path to the file to upload
        s3bucket (`str`): S3 bucket name
        s3_just_path (`str`): S3 path to file (without the s3://name/ prefix)

    Returns:
        None

    Raises:
        NoCredentialsError: if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
        PartialCredentialsError: if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are not valid
        S3UploadError: if there is an error uploading the file, could be S3 bucket not found
    """

    # Extract S3 bucket names from "s3://xxx"
    s3bucket = parse_s3(s3bucket)

    # connect to S3
    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    logger.info("AWS S3 bucket %s detected." % s3bucket)
    bucket = s3.Bucket(s3bucket)

    # Check if the file already exists in S3. If not, upload the file.
    try:
        logger.info("Checking whether file already exists in the S3 bucket", )
        s3.Object(s3bucket, s3_just_path).load()
    except NoCredentialsError as ne:
        logger.error(
            'Please provide AWS credentials via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables.')
        sys.exit(1)
    except PartialCredentialsError as pe:
        logger.error("Invalid AWS credentials")
        sys.exit(1)
    except ClientError as e:
        logger.info("File does not exist in S3. Uploading file to S3.")

        # Upload the file to specified S3 path
        try:
            bucket.upload_file(local_path, s3_just_path)
            logger.info('Data uploaded from %s to %s',
                        local_path, "s://"+s3bucket+"/"+s3_just_path)
        except boto3.exceptions.S3UploadFailedError as se:
            logger.error(
                """Upload failed. Please check if the file exists and if the S3 path is correct. % s""", se)
            sys.exit(1)
    else:
        logger.info("File already exists in S3. Skipping upload.")


def download_file_from_s3(local_path: str, s3bucket: str, s3_just_path: str) -> None:
    """Download file from S3 bucket

    Args:
        local_path (`str`): local path to download the file to.
        s3bucket (`str`): S3 bucket name
        s3_just_path (`str`): S3 path to file (without the s3://name/ prefix)

    Returns:
        None

    Raises:
        NoCredentialsError: if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
        PartialCredentialsError: if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are not valid
    """

    # Extract S3 bucket names from "s3://xxx"
    s3bucket = parse_s3(s3bucket)

    # connect to S3, raise errors if no valid AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in env
    try:
        s3 = boto3.resource('s3',
                            aws_access_key_id=AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        logger.info("S3 bucket %s connected.", s3bucket)
    except PartialCredentialsError as pe:
        logger.error("Invalid AWS credentials")
        raise pe
    except NoCredentialsError as ce:
        logger.error("""No AWS credentials detected in env. Please provide credentials via "
        "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables""")
        raise ce
    else:
        bucket = s3.Bucket(s3bucket)

    # Download the file from S3
    try:
        logger.info("Downloading from S3 bucket %s", s3bucket)
        bucket.download_file(s3_just_path, local_path)
    except NoCredentialsError as ne:
        logger.error(
            'Please provide AWS credentials via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables.')
        raise ne
    else:
        logger.info('Data successfully downloaded from %s to %s',
                    "s3://"+s3bucket+"/"+s3_just_path, local_path)
