# face-recognition-webcam
Recognise faces and and tag names on webcam video frames using face_recognition library


## Folder Structure

```
project
├───.metadata
├───logs
├───Workingfolder
│   ├───OutputVideo
│   └───TrainingImages
├───main.py
├───imageRecognition.py
└───logging_config.ini
```

1. `.metadata` will be use to store the faces_encoding

2. `Workingfolder` contains all the images needed to tag faces.

    - `TrainingImages` contains the images of people we want to use to train the model. Basically the people we want to tag later. It´s Important the name of the image is the name of the person in it. 


    - `OutputVideo` if the parameter `record = True` in the function imageRecognition.tagpeople_webcam then the recorded video will be stored in this folder by default.

## Usage 

1. Add into `Workingfolder/TrainingImages` one image for each person to tag. The name of the image is the name used later to tag that person. 

3. Run `main.py`
    - You can change the tolerance. The lower the tolerance more strict becomes the matching.  

## Colors

Depending on the distance there is a color code for the rectangles drawn. By default the tolerance is 0.6 and the colors are set as a proportion of the tolerance. Then by default this are the color codes:

- ![#606060](https://placehold.it/15/606060"/000000?text=+) `Grey`

```
distance > tolerance 
default: distance > 0.60
```

- ![#640000](https://placehold.it/15/640000"/000000?text=+) `Red`

```
distance > tolerance*0.84 
[default: distance >= 0.60*0.84 => distance >= 0.504]
```

- ![#ff8000](https://placehold.it/15/ff8000"/000000?text=+) `Orange`

```
distance >= tolerance*0.68
[default: distance >= 0.60*0.68 => distance >= 0.408]
```

- ![#006400](https://placehold.it/15/006400"/000000?text=+) `Green`

```
distance < tolerance*0.68
[default: distance < 0.60*0.68 => distance < 0.408]
```



## Credits 
- https://github.com/Lazymindz/MeetupMemberImageTag
- https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
- https://github.com/ageitgey/face_recognition
- https://github.com/davisking/dlib


