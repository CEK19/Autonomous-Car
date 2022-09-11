import numpy as np
import cv2
import time
import os

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
#------------------------- THRESHOLD FUNCTION--------------------------- #
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    # Return the combined s_channel & v_channel binary image
    return output

def s_channel_threshold(image, sthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]  # use S channel

    # create a copy and apply the threshold
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    return binary_output



def preprocessing(frame):    
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensitivity = 30
    lowerWhite = np.array([0,0,255-sensitivity])
    upperWhite = np.array([255,sensitivity,255])
    maskWhite = cv2.inRange(hsvImg, lowerWhite, upperWhite)
    
    lowerYellow = np.array([25,100 ,20]) #H,S,V
    upperYellow = np.array([32, 255,255]) #H,S,V
    maskYellow = cv2.inRange(hsvImg, lowerYellow, upperYellow)
    
    combineColorImg = cv2.bitwise_or(maskWhite, maskYellow)
    copyOriginalFrame = frame.copy()
    copyOriginalFrame[np.where(combineColorImg==[0])] = [0]
    
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)    
    cannyEdgeDectection = cv2.Canny(blurred, 200, 255)
    
    combineFirst = cv2.bitwise_or(cannyEdgeDectection, combineColorImg)
    
    backtorgb = cv2.cvtColor(combineFirst ,cv2.COLOR_GRAY2RGB)    
    return cv2.vconcat([backtorgb, frame])

def videoReading ():
    # VIDEO READING
    cap = cv2.VideoCapture("/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/dataset/real_video_Nhan/vid1.mp4")
    while(cap.isOpened()):
        begin = time.time()
        _, frame = cap.read()     
        begin = time.time()
        result = preprocessing(frame)        
        cv2.imshow("chac la ko gion dau", result)
    
        end = time.time()
        print(1/(end-begin))                
        key = cv2.waitKey(10)
        if (key == 32):
            cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    
def imageReading():
    # VIDEO READING
    PATH_IMAGE_INPUT = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/dataset/dataset_kitty/data_road/training/image_2/"
    imageCollections = os.listdir(PATH_IMAGE_INPUT)
    PATH_IMAGE_OUTPUT = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/report/10-09-2022-Nhan/training/" 
    
    for image in imageCollections:
        begin = time.time()
        frame = cv2.imread(PATH_IMAGE_INPUT + image, cv2.IMREAD_COLOR)        
        #------------------------- PREPROCESSING IMAGE ----------------------------
        resultPreprocessing = preprocessing(frame)
        cv2.imwrite(PATH_IMAGE_OUTPUT + image, resultPreprocessing)
        end = time.time()
        print(1/(end-begin))
        
# imageReading()
videoReading()
# cv2.destroyAllWindows()