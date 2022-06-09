# Identify MBTI Personality Type Based on Speech Texts

by Qingyang "Jojo" Zhou @jojozhouu

## Table of Contents

- [Project Charter](#Project-Charter)
- [Instructions for using the code](#Instructions-for-using-the-code)
  - [Uploading and downloading data to/from S3](#Uploading-and-downloading-data-to/from-S3)

## Project Charter

### Vision

Understanding their own personality helps people become more self-aware and emotionally intelligent. While people have unique personality traits, there are personality type indicator designed as a simple yet efficient way to encapsulate how people would behave, react, and prefer in different situations. While introspective self-report questionnaires are available for people to learn about their personality type, there are reasons for a text-based classifier. Firstly, it is widely acknowledged that [**self-report bias**](https://dictionary.apa.org/self-report-bias) could arise when people are asked to describe their thoughts, feelings, or behaviors rather then having them measured directly. It occurs when the respondents either don’t know the true answers or deliberately choose the options that are socially desired (e.g., a preferred personality type for an employment position they are applying for). Secondly, a text-based tool provides a faster, scalable, and practically easier way to identify personality type, especially of people who we don’t have direct connections with, such as celebrity or potential job hires. Given the maturity and popularity of social media, it is sometimes easier to collect posts written by someone, rather than having them fill out a questionnaire. For these reasons, this project aims to create a simple, self-service tool that anyone could use to identify personality type based on texts.

### Mission

The goal of the project is to identify Myers-Briggs Type Indicator (MBTI) personality type based on speech texts. The MBTI type describes personalities on four dimensions, with each dimension being a dichotomy between 2 possible types.

- **Favorite world**: Extraversion (E) vs. Introversion (I)
- **Information**: Sensing (S) vs. Intuition (N)
- **Decisions**: Thinking (T) vs. Feeling (F)
- **Structure**: Judging (J) vs. Perceiving (P)

More information of the personality type classification could be obtained at [**Myers-Briggs’s website**](https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/).

To use the model, the users will input texts to the model, such as social media posts and conversation transcripts. The model will then classify the speaker of the given texts into one of the sixteen possible MBTI personality types, based solely on the content of the texts. The dataset to be used in model building is collected from Kaggle, under the title “[**(MBTI) Myers-Briggs Personality Type Dataset**](https://www.kaggle.com/datasets/datasnaek/mbti-type)” by MITCHELL J. The datasets contain the content of more than 9K posts collected from PersonalityCafe forum and the MBTI types of the authors.

### Success Criteria

There are two types of sucess metrics.

- **Model Performance Metric** \
  The model performance metric is the _averaged F1-score_ across all the dichotomous pairs. Under the assumption that each dichotomous pair is class-balanced, the model will be deployed if it can reach an averaged F1-score of 0.7. However, it is likely that the classes are imbalanced, which would affect F1 scores. Then, the F1-score threshold will be re-decided based on the degree of imbalance.

- **Business Metrics** \
  The business metrics to be measured in the long term will be related to user engagement. It is expected that if users believe the model is predicting their types accurately, they will share the links to their friends and re-use the tool more frequently. Thus, the two metrics will be _number of distinct users using the tool per week_ (measuring popularity) and _average number of visits per user_ (measuring stickiness).


## Instructions for using the code

Before using the code, please make sure the following environmental variables are already sourced.
- SQLALCHEMY_DATABASE_URI
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- S3_BUCKET
 
Additionally, please make sure Northwestern VPN is connected.

### Uploading and downloading data to/from S3
1. Build a docker image called `final-project` for managing database connection, managing S3, and running the model pipeline. If the image has been built from previous step, skip this step.

   ```bash
   docker build -f dockerfiles/Dockerfile -t final-project .
   ```
2. To upload a local file to S3, run the following command. Uploads will be performed if the file does not exist in the S3 bucket specified.

   ```bash
   docker run -e S3_BUCKET -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY --mount type=bind,source="$(pwd)"/,target=/app/ final-project run.py manage_s3 upload_data
   ```

### Creating database and ingesting data to RDS or SQLite database
1. Build a docker image called `final-project` for managing database connection, managing S3, and running the model pipeline. If the image has been built from previous step, skip this step.

   ```bash
   docker build -f dockerfiles/Dockerfile -t final-project .
   ```
2. To create a database at the given SQLALCHEMY_DATABASE_URI, run the following command.
   
   ```bash
   docker run -e SQLALCHEMY_DATABASE_URI  --mount type=bind,source="$(pwd)"/,target=/app/ final-project run.py manage_rds create_db
   ```
3. To ingest the raw data file to the given SQLALCHEMY_DATABASE_URI, run the following command.

   ```bash
   docker run -e SQLALCHEMY_DATABASE_URI  --mount type=bind,source="$(pwd)"/,target=/app/ final-project run.py manage_rds ingest_data
   ```
4. To delete the database at the given SQLALCHEMY_DATABASE_URI, run the following command.
   
   ```bash
   docker run -e SQLALCHEMY_DATABASE_URI  --mount type=bind,source="$(pwd)"/,target=/app/ final-project run.py manage_rds delete_db
   ```

### Model training pipeline (run.py)

1. Build a docker image called `final-project` for managing database connection, managing S3, and running the model pipeline. If the image has been built from previous step, skip this step.

   ```bash
   docker build -f dockerfiles/Dockerfile -t final-project .
   ```

2. To run each step separately, run the following docker command. The last argument could be changed to perform different steps, but the arguments need to be run in order.
    - acquiring data from S3: `acquire`
    - cleaning texts data: `clean`
    - training models: `train`
    - predicting test set or new texts data: `predict`
    - evaluating on test set: `evaluate`
 
   ```bash
   docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)"/data,target=/app/data final-project model <step>
   ```
   
   For example, to acquire data, the complete docker command will be:
   ```bash
   docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)"/data,target=/app/data final-project model acquire
   ```

3. To run the entire model training pipeline at once, run the following docker command.
    
   ```bash
   docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)"/data,target=/app/data final-project model all
   ```

### Unit tests

1. Build a docker image called `final-project-tests` to run unit tests on the `/src` scripts, `run.py`, and `app.py`.
    
    ```bash
    docker build -f dockerfiles/Dockerfile.test -t final-project-tests .
    ```
2. To run the tests in the `/tests` folder, run the following command.

    ```bash
    docker run final-project-tests
    ```

### Using the Flask app (app.py)

1. Build docker image called `final-project-app` to use the web app via docker.
    
    ```bash
    docker build -f dockerfiles/Dockerfile.app -t final-project-app .
    ```
2. To run the web app, run the following command, and then enter address as "localhost:5000" in the browser. For example, on Windows it will be "http://127.0.0.1:5000/"

    ```
    docker run -e SQLALCHEMY_DATABASE_URI -p 5000:5000 final-project-app
    ```
