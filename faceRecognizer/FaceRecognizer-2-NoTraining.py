import face_recognition
import cv2
print(cv2.__version__)

donFace = face_recognition.load_image_file('/home/tareeq/Desktop/PyPro/FaceRecognizer/demoImages/known/Donald Trump.jpg')
donEncode = face_recognition.face_encodings(donFace)[0]

nancyFace = face_recognition.load_image_file('/home/tareeq/Desktop/PyPro/FaceRecognizer/demoImages/known/Nancy Pelosi.jpg')
nancyEncode = face_recognition.face_encodings(nancyFace)[0]

mikeFace = face_recognition.load_image_file('/home/tareeq/Desktop/PyPro/FaceRecognizer/demoImages/known/Mike Pence.jpg')
mikeEncode = face_recognition.face_encodings(mikeFace)[0]

Encodings = [donEncode, nancyEncode, mikeEncode]
Names = ['The Donald', 'Nancy Pelosi', 'Mike Pence']

font = cv2.FONT_HERSHEY_SIMPLEX
testImage = face_recognition.load_image_file('/home/tareeq/Desktop/PyPro/FaceRecognizer/demoImages/unknown/u11.jpg')

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



"""
image = face_recognition.load_image_file('/home/tareeq/Desktop/PyPro/FaceRecognizer/demoImages/unknown/u3.jpg')
face_locations = face_recognition.face_locations(image)
print(face_locations)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for(row1,col1,row2,col2) in face_locations:
    cv2.rectangle(image, (col1,row1), (col2,row2), (0,0,255),2)

cv2.imshow('myWindow',image)
cv2.moveWindow('myWindow',0,0)

if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()

"""