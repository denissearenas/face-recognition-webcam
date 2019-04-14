import face_recognition
import time
import os, os.path
import numpy as np
import cv2

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.
# Code skeleton from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

# Load a sample picture and learn how to recognize it.
def createAndLoadEncodings(TrainingFolder):
    loaded_face_encodings = {}
    if len(os.listdir(TrainingFolder)) > 0:
        for file in os.listdir(TrainingFolder):
            name = file
            if file.rfind('.') >= 0:
                #remove file extension from name
                name = file[:file.rfind('.')]
            name_image = face_recognition.load_image_file(os.path.join(TrainingFolder,file))
            name_face_encoding = face_recognition.face_encodings(name_image)
            if len(name_face_encoding) > 0:
                loaded_face_encodings[name] = name_face_encoding[0]
    np.save('./.metadata/faces_encoding.npy', loaded_face_encodings)
    return np.load('./.metadata/faces_encoding.npy')

def loadEncodings(TrainingFolder,retrain=False):
    if (not os.path.exists('./.metadata/faces_encoding.npy')) or (retrain==True):
        return createAndLoadEncodings(TrainingFolder)
    else:
        return np.load('./.metadata/faces_encoding.npy')


def tagpeople_webcam( loaded_face_encodings, tolerance=0.60, record = False, skip_frame = True, output_folder = ''):
    video_capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    process_this_frame = True    

    known_face_names, known_face_encodings = list(loaded_face_encodings[()].keys()), list(loaded_face_encodings[()].values())

    if record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestring = time.strftime("%Y%m%d_%H%M%S")
        out = cv2.VideoWriter( output_folder + "output_"+timestring+".avi",fourcc, 4.0, (640,480))

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
    
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
    
        # Only process every other frame of video to save time
        if process_this_frame:            

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for  face_encoding in  face_encodings:

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                name = "Unknown"
                color = (96, 96, 96) #grey
                min_face_distances = min(face_distances)
                distance = str(round(min_face_distances,3))

                #% of tolerance to change color:
                tolerance_red = tolerance*0.85
                tolerance_orange = tolerance*0.67

                # If a match was found in known_face_encodings, use the one with the minimun distance 
                if min_face_distances <= tolerance:
                    match_index = np.argmin(face_distances)
                   # match_index = face_distances.index(min(face_distances))
                    name = known_face_names[match_index]
                    if min(face_distances) >= tolerance_red:
                        color = (0, 0, 100)
                    elif min(face_distances) >= tolerance_orange:
                        color = (0, 128, 255)
                    else:
                        color = (0,100,0)
                    name =  name + ' - ' + distance    
    
                    face_names.append(name)

        if skip_frame:
            process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
                                
            # Draw a box around the face
            rectangle_width = 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, rectangle_width)
            
            #get the text width and height for the name, font and font_scale chosen
            size = cv2.getTextSize(name, font, font_scale, 1)
            text_width = size[0][0]
            text_height = size[0][1]
            
            #offset: extra lengh needed in each side to fit the name.
            offset = 0
            #If Label is longer than rectangle width, calculate offset needed.
            if ( right-left ) < text_width :
                offset =  int((text_width - ( right - left ))/2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left-offset-int(rectangle_width/2), bottom + int(text_height*2) ), (right+offset+int(rectangle_width/2), bottom), color, cv2.FILLED)
            
            cv2.putText(frame, name, (left -offset, bottom + int(text_height*2) - 6), font, font_scale, (255, 255, 255), 1)
        
        cv2.imshow('Video', frame)

        if record:
            out.write(frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
 
    # Release handle to the webcam
    video_capture.release()
    if record:
        out.release()
    cv2.destroyAllWindows()

