import face_recognition
import cv2
import os
import pickle
print(cv2.__version__)

j = 0

Encodings = []
Names =[]

with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX

image_dir_u = '/home/tareeq/Desktop/PyPro/FaceRecognizer/demoImages/unknown/'

for root,dirs, files in os.walk(image_dir_u):
    for file in files:
        testImagePath = os.path.join(root,file)
        print(testImagePath)
        testImage = face_recognition.load_image_file(testImagePath)
        facePositions = face_recognition.face_locations(testImage)
        allEncodings = face_recognition.face_encodings(testImage,facePositions)
        testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)

        for (top,right,bottom,left), face_encoding in zip(facePositions, allEncodings):
            name = 'Unknown Person'
            matches =  face_recognition.compare_faces(Encodings,face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = Names[first_match_index]
            cv2.rectangle(testImage,(left,top), (right,bottom),(0,0,255),2)
            cv2.putText(testImage, name, (left, top-6), font, 1, (0,255,255),1)

        cv2.imshow('myWindow',testImage)
        cv2.moveWindow('myWindow',0,0)

        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()
