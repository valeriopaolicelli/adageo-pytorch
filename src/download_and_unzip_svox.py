
import os
import shutil
import requests

def download_file_from_google_drive(zip_destination="svox.zip",
                                    file_id="16iuk8voW65GaywNUQlWAbDt6HZzAJ_t9"):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    def save_response_content(response, zip_destination):
        CHUNK_SIZE = 32768
        with open(zip_destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, zip_destination)    

def download_and_unzip_svox(dataset_destination="datasets",
                            zip_destination="svox.zip", delete_zip=True):
    download_file_from_google_drive(zip_destination=zip_destination)
    shutil.unpack_archive(zip_destination, dataset_destination)
    if delete_zip:
        os.remove(zip_destination)

if __name__ == "__main__":
    download_and_unzip_svox()

