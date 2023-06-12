import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.layers import *
from keras.optimizers import *
import numpy as np
import cv2
import random
from keras.models import *
import sys
import copy

from interpolation import interpolate

dimImage = 128

g_alpha,g_gamma = interpolate()
# print(g_alpha,g_gamma)
# g_alpha,g_gamma = 0.7,0.75

class stupidModel:
    def unet(self, input_size, loss, metrics):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
        
        self.model = Model(inputs = inputs, outputs = conv10)
        self.model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics =[metrics])
        # self.model.compile(optimizer = Adam(lr = 3e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return self.model

def tversky(y_true, y_pred):
    smooth = 0.01
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    global g_alpha
    return (true_pos + smooth)/(true_pos + g_alpha*false_neg + (1-g_alpha)*false_pos + smooth)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    global g_gamma
    return K.pow((1-pt_1), g_gamma)
def dice(y_true, y_pred):
    smooth = 0.01
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
def rotate_image(image, alpha):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, alpha, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def ImageArgument(frame,brightnessOffset,noise,blur):
    frame = np.array(frame,dtype="int")
    frame = np.clip(frame - brightnessOffset,0,255)
    for j in range(int(100*noise)):
        frame[int(random.random()*128)][int(random.random()*128)] = random.random()*255
    frame = np.array(frame,dtype="uint8")
    frame = cv2.blur(frame, (blur,blur)) 
    return frame

def randomize_rect(image, x, y, width, height):
    # Tạo một bản sao của hình ảnh đầu vào để tránh ảnh hưởng đến hình ảnh gốc
    img_copy = np.copy(image)

    # Tạo một hình chữ nhật ngẫu nhiên với kích thước được xác định bởi đầu vào (x,y,width,height)
    rect = img_copy[y:y+height, x:x+width]

    # Tạo các giá trị ngẫu nhiên cho các pixel trong hình chữ nhật
    random_values = np.random.randint(low=0, high=256, size=rect.shape)

    # Gán các giá trị ngẫu nhiên vào các pixel trong hình chữ nhật
    rect[:] = random_values

    # Trả về hình ảnh được cập nhật
    return img_copy

def augment_image(image):
    # Randomly apply Gaussian blur
    rotateVal = np.random.randint(-30,30)
    image = rotate_image(image,rotateVal)
    blur_radius = random.randint(0, 3)
    if blur_radius > 0:
        image = cv2.GaussianBlur(image, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)

    # Randomly adjust brightness
    brightness = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    dark = random.randint(-150, 10)
    lut = np.arange(0, 256, 1, dtype=np.int32)
    lut = np.clip(lut + dark, 0, 255).astype(np.uint8)
    image = cv2.LUT(image, lut)

    # Randomly add frost effect
    frost_level = random.uniform(0, 0.2)
    noise = np.random.randn(*image.shape) * 50
    noise = np.clip(noise, -255, 255).astype(np.uint8)
    image = cv2.addWeighted(image, 1 - frost_level, noise, frost_level, 0)

    # Randomly hide some part of the image
    x,y,w,h = np.random.randint(80),np.random.randint(80),np.random.randint(47),np.random.randint(47)
    image = randomize_rect(image,x,y,w,h)
    return image,[x,y,w,h],rotateVal

stpm = stupidModel()

# cnn = stpm.unet(input_size=(dimImage,dimImage,1),loss=focal_tversky,metrics=dice)
workingDir  = "D:/container/AI_DCLV/21-4"
testDir = "D:/container/AI_DCLV/readData/test_v1"
dataLink = "D:/container/AI_DCLV/readData/labeled_v7"
backgroundLink = "D:/container/AI_DCLV/readData"

# cnn = load_model("D:/container/AI_DCLV/21-4/Model-S1_AUTO-"+str(len(os.listdir(workingDir)))+".hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})
cnn = load_model("D:/container/AI_DCLV/27-4/Model-S3_AUTO-0.hdf5",custom_objects={'focal_tversky':focal_tversky, 'dice':dice})
# cnn = stpm.unet(input_size=(dimImage,dimImage,1),loss=focal_tversky,metrics=dice)

dataX = []
dataY = []

backgroundImgList = os.listdir(backgroundLink)

nameList = os.listdir(dataLink+"/image")
print("read image")
for eachName in nameList:
    if (not os.path.exists(dataLink+"/label/" + eachName)):
        print("skip ",dataLink+"/label/" + eachName)
        continue
    tmpX = cv2.imread(dataLink+"/image/" + eachName)
    tmpX = cv2.resize(tmpX,(dimImage,dimImage))
    tmpX = cv2.cvtColor(tmpX,cv2.COLOR_RGB2GRAY)
    dataX.append(np.array(tmpX))
    tmpX = cv2.flip(tmpX,1)
    dataX.append(np.array(tmpX))

    tmpY = cv2.imread(dataLink+"/label/" + eachName)
    tmpY = cv2.resize(tmpY,(dimImage,dimImage))
    tmpY = cv2.cvtColor(tmpY,cv2.COLOR_RGB2GRAY)
    _,tmpY = cv2.threshold(tmpY,1,255,cv2.THRESH_BINARY)
    dataY.append(np.array(tmpY))
    tmpY = cv2.flip(tmpY,1)
    dataY.append(np.array(tmpY))

validX = []
validY = []
for eachName in os.listdir(testDir+"/image"):
    tmpX = cv2.imread(testDir+"/image/" + eachName)
    tmpX = cv2.resize(tmpX,(dimImage,dimImage))
    tmpX = cv2.cvtColor(tmpX,cv2.COLOR_RGB2GRAY)
    validX.append(np.array(tmpX))

    tmpY = cv2.imread(testDir+"/label/" + eachName)
    tmpY = cv2.resize(tmpY,(dimImage,dimImage))
    tmpY = cv2.cvtColor(tmpY,cv2.COLOR_RGB2GRAY)
    _,tmpY = cv2.threshold(tmpY,1,255,cv2.THRESH_BINARY)
    validY.append(np.array(tmpY))

print("read done")
listDataX = []
listDataY = []

for j in range(1024):
    if (np.random.randint(10) < 8):
        index = np.random.randint(len(dataX))

        p1 = (np.random.randint(dimImage),np.random.randint(dimImage))
        p2 = (np.random.randint(dimImage),np.random.randint(dimImage))
        thickness = np.random.randint(1,5)

        xLine = cv2.line(copy.deepcopy(dataX[index]),p1,p2,[np.random.randint(255)]*3,thickness,cv2.LINE_AA)
        yLine = cv2.line(copy.deepcopy(dataY[index]),p1,p2,0,thickness,cv2.LINE_AA)

        tmpX,rectPos,rotateVal = augment_image(xLine)
        tmpY = rotate_image(yLine,rotateVal)
        tmpY = cv2.rectangle(tmpY,(rectPos[0],rectPos[1]),(rectPos[0]+rectPos[2],rectPos[1]+rectPos[3]),0,-1)
        listDataX.append(tmpX)
        listDataY.append(tmpY)
    else:
        frame = cv2.imread(backgroundLink+"/"+backgroundImgList[np.random.randint(len(backgroundImgList))])
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,(dimImage,dimImage))
        tmpX,_,_ = augment_image(frame)
        listDataX.append(tmpX)
        listDataY.append(np.zeros((dimImage,dimImage),dtype="uint8"))
        # tmpY = np.zeros((dimImage,dimImage),dtype="uint8")

    # tmpX = cv2.hconcat([tmpX,tmpY])
    # tmpX = cv2.resize(tmpX,(512*2,512))
    # cv2.imshow("f",tmpX)
    # if cv2.waitKey(0) == 27:
    #     sys.exit()

listDataX = np.array(listDataX,dtype="float32")
listDataY = np.array(listDataY,dtype="float32")

validX = np.array(validX,dtype="float32")
validY = np.array(validY,dtype="float32")

listDataY /= 255
validY /= 255

history = cnn.fit(listDataX, listDataY, batch_size = 16, epochs = 4,validation_data=(validX,validY))
# cnn.save("D:/container/AI_DCLV/27-4/Model-S3_AUTO-2.hdf5")
f = open("D:/NDT/PY_ws/DCLV/src/log.txt", "a")
f.write(str(g_alpha) + " " + str(g_gamma) + " " + str(history.history["dice"][-1]) + " " + str(history.history["val_dice"][-1])+"\n")
f.close()
print("DONE")
