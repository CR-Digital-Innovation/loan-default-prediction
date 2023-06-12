"""Useful S3 operations to perform various operations involves with csv files, data frames, pickle files using s3fs library"""

import s3fs
import pandas as pd
import joblib


class S3Utils:
    def __init__(
        self, aws_access_key: str, aws_secret_key: str, bucket_name: str, data_dir: str
    ) -> None:
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.s3_session = s3fs.S3FileSystem(
            key=self.aws_access_key, secret=self.aws_secret_key
        )

    def get_s3_path(self, dirPath: str, fileName: str) -> str:
        """Simple function to return an S3 Path"""
        path = f"s3://{self.bucket_name}/{self.data_dir}/{dirPath}/{fileName}"
        return path

    def load_dataframe(self, dirPath: str, fileName: str) -> pd.DataFrame:
        """Function to load csv/tsv file from AWS S3 bucket and retun a pandas dataframe"""

        try:
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Loading '{file_path}' file as dataframe.")
            with self.s3_session.open(file_path, "rb") as file:
                dataframe = pd.read_csv(file)
            return dataframe
        except Exception as e:
            print(f"Error reading csv data as dataframe from S3: {e}")
            return None

    def save_dataframe(self, dirPath: str, fileName: str, df: pd.DataFrame) -> None:
        """Function to write a dataframe as csv file to the S3 storage"""

        try:
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Saving dataframe as '{fileName}' at '{file_path}'")
            with self.s3_session.open(file_path, "w") as file:
                df.to_csv(file, index=False)

            # Check if file is saved in S3 bucket, return success if exists.
            file_exists = self.s3_session.exists(file_path)
            if file_exists:
                print(f"csv file '{fileName}' is saved to S3 successfully.")
        except Exception as e:
            print(f"Error saving dataframe as csv to S3: {e}")

    def save_pickle(self, dirPath: str, fileName: str, data: any) -> None:
        """Function to save pickle file to S3 storage using joblib"""

        try:
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Saving pickle file '{fileName}' at '{file_path}'")
            with self.s3_session.open(file_path, "wb") as file:
                joblib.dump(data, file)

            # Check if file is saved in S3 bucket, return success if exists.
            file_exists = self.s3_session.exists(file_path)
            if file_exists:
                print(f"Pickle file '{fileName}' saved to S3 successfully.")
        except Exception as e:
            print(f"Error saving pickle file to S3: {e}")

    def load_pickle(self, dirPath: str, fileName: str) -> any:
        """Function to load pickel file from S3 storage using joblib"""

        try:
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Loading pickle file from '{file_path}'")
            with self.s3_session.open(file_path, "rb") as file:
                data = joblib.load(file)
            return data
        except Exception as e:
            print(f"Error loading pickle file from S3 storage: {e}")
            return None

    def check_file_exist(self, dirPath: str, fileName: str) -> bool:
        """Function to check if a file exist in given path in S3"""

        try:
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Checking for the file {file_path}")
            
            # Checking the file
            file_exist = self.s3_session.exists(file_path)
            if file_exist:
                print(f"File {file_path} exist")
            else:
                print(f"File {file_path} does not exist")
            return file_exist
        except Exception as e:
            print(f"Error checking the file in S3: {e}")
