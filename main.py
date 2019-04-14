import logging
from logging.config import fileConfig

import os, os.path

import imageRecognition

TrainingFolder = 'WorkingFolder/TrainingImages/'
OutputFolder = 'WorkingFolder/OutputVideo/'

# Create the Working folders
working_folders = ['logs','.metadata','./WorkingFolder/OutputVideo/']
[os.makedirs(folder) for folder in working_folders if not os.path.exists(folder)]

# Load log config
fileConfig('logging_config.ini')
logger = logging.getLogger()


if __name__ == "__main__":
    
    #video_capture = cv2.VideoCapture(0)
    
    encodings = imageRecognition.loadEncodings(TrainingFolder)

    imageRecognition.tagpeople_webcam( encodings, record = False, skip_frame = True, output_folder = OutputFolder)

