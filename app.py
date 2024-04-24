from main import app
from waitress import serve

import requests
import os


def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a file in binary write mode and write the content of the response
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File '{filename}' downloaded successfully!")
    else:
        print(f"Failed to download file from '{url}'. Status code: {response.status_code}")


def execDownloadFile():
    # URL of the file to download
    url1 = "https://pjreddie.com/media/files/yolov3.weights"
    # File name to save the downloaded file
    filename1 = "yolov3.weights"
    # Download the file into the project folder
    download_file(url1, filename1)

    # URL of the file to download
    url2 = "https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg"
    # File name to save the downloaded file
    filename2 = "yolov3.cfg"
    # Download the file into the project folder
    download_file(url2, filename2)

    # URL of the file to download
    url3 = "https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names"
    # File name to save the downloaded file
    filename3 = "coco.names"
    # Download the file into the project folder
    download_file(url3, filename3)

if __name__ == '__main__':
    # app.run()
    execDownloadFile()
    serve(app, host='0.0.0.0', port=5000)