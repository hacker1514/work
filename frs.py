#face recognizer by Niranjan

from cv2 import imread
from numpy import array,argmax,resize
from tensorflow.keras import layers, Sequential

model = Sequential(
    [
    layers.Input(shape=(64,64,1)),
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(2,activation="softmax")]
)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

x = []
y = []

for i in range(1,21):
  img = imread("data/"+str(i)+".jpg",0)
  img = resize(img,(64,64))
  x.append(img)

y=array([[1]]*10 + [[0]]*10)

x = array(x).reshape(-1,64,64,1)/255.0

model.fit(x,y,epochs=100,batch_size=4)

def check(img):
    x = imread(img,0)
    x = resize(x,(64,64)).reshape(1,64,64,1)
    x  = x/255.0
    return argmax(model.predict([x],verbose=0)[0])