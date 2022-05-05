import os.path

from src.load_save_data_s3 import upload_file_to_s3, download_file_from_s3

# TODO logging.config.fileConfig('config/logging/local.conf')
# TODO logger = logging.getLogger('penny-lane-pipeline')

if __name__ == '__main__':

    # Upload raw data files to given S3 bucket
    # upload_file_to_s3("data/raw/raw_forum.csv",
    #                  "s3://2022-msia423-zhou-jojo/raw/raw_forum.csv")
