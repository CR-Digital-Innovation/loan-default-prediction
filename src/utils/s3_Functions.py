"""Useful S3 operations to perform various operations involves with csv files, data frames, pickle files using s3fs library"""
import io
import gzip
import s3fs
import pandas as pd
import joblib
from tqdm import tqdm


# Define a custom S3Utils class for S3 functions
class S3Utils:
    """A Class to define various methods for different S3 operations"""

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

    def get_file_size(self, filePath: str) -> int:
        """Function to get the file size of a file in S3"""

        file_info = self.s3_session.info(filePath)
        return file_info["size"]

    def load_dataframe(self, dirPath: str, fileName: str) -> pd.DataFrame:
        """Function to load csv/tsv file from AWS S3 bucket and retun a pandas dataframe"""

        try:
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Loading '{file_path}' file as dataframe.")

            # Define a custom block size for reading the file
            block_size = 8192

            progress_bar = tqdm(
                total=self.get_file_size(file_path),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Loading",
            )

            # Open the file using the custom BlockReader and load with joblib
            with self.s3_session.open(
                file_path,
                "rb",
            ) as file:
                chunks = []
                while True:
                    chunk = file.read(block_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()

            csv_data = b"".join(chunks).decode()
            dataframe = pd.read_csv(io.StringIO(csv_data))
            return dataframe
        except Exception as error:
            print(f"Error reading csv data as dataframe from S3: {error}")
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
        except Exception as error:
            print(f"Error saving dataframe as csv to S3: {error}")

    def save_pickle(
        self, dirPath: str, fileName: str, data: any, compress: bool = True
    ) -> None:
        """Function to save pickle file to S3 storage using joblib"""

        try:
            fileName = f"{fileName}.gz" if compress else fileName
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Saving pickle file '{fileName}' at '{file_path}'")

            if compress:
                with self.s3_session.open(file_path, "wb") as file:
                    with gzip.GzipFile(
                        fileobj=file, mode="wb", compresslevel=6
                    ) as gz_file:
                        joblib.dump(data, gz_file)
            else:
                with self.s3_session.open(file_path, "wb") as file:
                    joblib.dump(data, file)

            # Check if file is saved in S3 bucket, return success if exists.
            file_exists = self.s3_session.exists(file_path)
            if file_exists:
                print(f"Pickle file '{fileName}' saved to S3 successfully.")
        except Exception as error:
            print(f"Error saving pickle file to S3: {error}")

    def load_pickle(self, dirPath: str, fileName: str, compressed: bool = True) -> any:
        """Function to load pickel file from S3 storage using joblib"""

        try:
            fileName = f"{fileName}.gz" if compressed else fileName
            file_path = self.get_s3_path(dirPath, fileName)
            print(f"Loading pickle file from '{file_path}'")

            # Define a custom block size for reading the file
            block_size = 8192

            progress_bar = tqdm(
                total=self.get_file_size(file_path),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Loading",
            )

            # Open the file using the custom BlockReader and load with joblib
            with self.s3_session.open(
                file_path,
                "rb",
            ) as file:
                chunks = []
                while True:
                    chunk = file.read(block_size)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()

            file_obj = b"".join(chunks)

            if compressed:
                with gzip.GzipFile(fileobj=io.BytesIO(file_obj), mode="rb") as gz_file:
                    data = joblib.load(gz_file)
            else:
                data = joblib.load(io.BytesIO(file_obj))
            return data
        except Exception as error:
            print(f"Error loading pickle file from S3 storage: {error}")
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
        except Exception as error:
            print(f"Error checking the file in S3: {error}")
