import os.path

from src.load_save_data_s3 import extract_7z, upload_file_to_s3

# TODO logging.config.fileConfig('config/logging/local.conf')
# TODO logger = logging.getLogger('penny-lane-pipeline')

if __name__ == '__main__':

    # Extract compressed raw data files if not yet extracted
    if not os.path.isfile("data/raw/raw_500word_processed.csv"):
        raw_500word = extract_7z(
            "data/raw/raw_500word_processed.7z", "data/raw/")

    # Upload raw data files to given S3 bucket
    # upload_file_to_s3("data/raw/raw_500word_processed.csv",
    #                  "s3://2022-msia423-zhou-jojo/raw/raw_500word_processed.csv")
    # upload_file_to_s3("data/raw/raw_forum.csv",
    #                  "s3://2022-msia423-zhou-jojo/raw/raw_forum.csv")
