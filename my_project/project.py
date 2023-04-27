import cv2

# pre-trained face detection
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# taking image
img = cv2.imread("RDJ.jpg")

# converts img to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detects faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# shows image
cv2.imshow("Face Detector", grayscaled_img)
cv2.waitKey()

















