import sys, os
import pandas as pd
import boto3
import botocore

def s3_object_exists(bucket_name, 
                     object_key):
    """
    ENSURE OBJECT KEY EXISTS IN S3 BUCKET
    """
    key_exists = True
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket_name, object_key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            key_exists = False
        else:
            raise e
    return  key_exists

def download_from_s3(bucket_name, 
                     object_key, 
                     local_file_path):
    """
    DOWNLOAD DATA FROM A PUBLIC S3 BUCKET
    """
    s3 = boto3.resource('s3')

    ### CHECK OBJECT EXISTS
    data_presence = s3_object_exists(bucket_name, object_key)
    if not data_presence:
        print(f"Data was not found to exist in S3.")
        return False

    ### CHECK OBJECT IS NOT ALREADY LOCAL
    if os.path.isfile(local_file_path):
        print(f"{local_file_path} already exists. Skipping download.")
        return False

    ### ATTEMPT DOWNLOAD FROM S3
    try:
        print(f"Downloading data export from the Dharma Overground...")
        s3.Bucket(bucket_name).download_file(object_key, local_file_path)
        print(f"Successfully downloaded {object_key} to {local_file_path}")
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            print(f"An error occurred: {e}")
        return False


