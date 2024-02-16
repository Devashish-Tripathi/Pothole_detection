#This code uses the Roboflow model and the API key issued to run the visualization

from roboflow import Roboflow
import cv2
rf = Roboflow(api_key="vftys1pnN2w8wiKHPshs")
project = rf.workspace().project("pothole-detection-vy8up")
model = project.version(1).model

img='many_potholes.jpeg'
pred=f'pred_{img}'

# infer on a local image
print(model.predict(img, confidence=40, overlap=30).json())

# visualize your prediction
model.predict(img, confidence=40, overlap=30).save(pred)

og=cv2.imshow("Original",cv2.imread(img))
cv2.waitKey(0)
pred=cv2.imshow("Predicted",cv2.imread(pred))
cv2.waitKey(0)
