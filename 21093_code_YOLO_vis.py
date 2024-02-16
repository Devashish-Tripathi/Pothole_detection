#This code used Roboflow server to visualize the YOLO results by uploading the trained weights
#Due to GPU and TensorFlow issues, Colab was not allowing the weights to be used in itself, as such this approach had to be taken

from roboflow import Roboflow
rf = Roboflow(api_key="vftys1pnN2w8wiKHPshs")
project = rf.workspace().project("pothole-detection-2-28w0d")
model = project.version(1).model

# infer on a local image
print(model.predict("many_potholes.jpeg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("many_potholes.jpeg", confidence=40, overlap=30).save("img_yolo.jpeg")
