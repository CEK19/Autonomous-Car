import numpy as np
import cv2
import time
import os

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*2.85/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:            
            for x1, y1, x2, y2 in line:                
                line_image = cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)                                        
    return line_image

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
 
#------------------------- REGION OF INTEREST FUNCTION --------------------------- #
def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]

    # IMAGE FULL BLACK 
    mask = np.zeros_like(canny)    

    # triangle = np.array([[
    # (200, height),
    # (550, 250),
    # (1100, height),]], np.int64)    

    triangle = np.array([[0, 320], [240, 180], [800, 180], [canny.shape[1], 320]])           
    cv2.fillPoly(mask, pts=[triangle], color =(255,255,255))    
    masked_image = cv2.bitwise_and(canny, mask)    
    cv2.circle(masked_image, (0, 320), 20, (255, 0, 0), 2)
    cv2.circle(masked_image, (240, 180), 20, (255, 0, 0), 2)
    cv2.circle(masked_image, (800, 180), 20, (255, 0, 0), 2)
    cv2.circle(masked_image, (canny.shape[1], 320), 20, (255, 0, 0), 2)
    return masked_image

#------------------------- FINDING LANE USING SLIDE WINDOW --------------------------- #
def find_lane_pixels_using_histogram(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 10 # 100 50
    # Set minimum number of pixels found to recenter window
    minpix = 30

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def fit_poly(binary_warped,leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty):     
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
        
    margin = 100
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='red')
    ## End visualization steps ##
    return result


def videoReading ():
    # VIDEO READING
    cap = cv2.VideoCapture("map_4_lane_ngoai.avi")
    while(cap.isOpened()):
        begin = time.time()
        _, frame = cap.read()     
        begin = time.time()
        #------------------------- PREPROCESSING IMAGE ----------------------------
        gradx = abs_sobel_thresh(frame, orient='x', thresh_min=60, thresh_max=255) # (20, 100)
        grady = abs_sobel_thresh(frame, orient='y', thresh_min=60, thresh_max=255) # (20, 100)
        c_binary = color_threshold(frame, sthresh=(100,255), vthresh=(50,255)) # GET BINARY

        preprocessImage = np.zeros_like(frame[:,:,0]) 
        preprocessImage[((gradx == 1) & (grady ==1) | (c_binary == 1))] = 255  # FILTER
        # canny = cv2.Canny(frame, 50, 150, apertureSize=3, L2gradient=True) -- JUST FOR COMPARTING 
        cv2.imshow("title", frame)        
        
        #------------------------- CREATE REGION OF INTEREST ---------------------------
        bitwiseImg = region_of_interest(preprocessImage)
        cv2.imshow("bitwise", bitwiseImg)
        
        #------------------------- WARP PERPESTIVE ---------------------------    
        # src_bot_left = [0, 320]
        # src_top_left = [240, 255]
        # src_top_right = [425, 255]
        # src_bot_right = [frame.shape[1], 320]
        
        # dst_bot_left = [src_bot_left[0]/2 + src_top_left[0]/2, src_bot_left[1] + 50]
        # dst_top_left = [src_bot_left[0]/2 + src_top_left[0]/2, src_top_left[1] - 50]
        # dst_top_right = [src_bot_right[0]/2 + src_top_right[0]/2, src_top_right[1] - 50]
        # dst_bot_right = [src_bot_right[0]/2 + src_top_right[0]/2, src_bot_right[1] + 50]
            
        # src = np.float32([src_bot_left, src_top_left, src_top_right, src_bot_right])
        # dst = np.float32([dst_bot_left, dst_top_left, dst_top_right, dst_bot_right])
        
        # M = cv2.getPerspectiveTransform(src,dst)
        # Minv = cv2.getPerspectiveTransform(dst,src)
        # binary_warped = cv2.warpPerspective(bitwiseImg,M,(frame.shape[1], frame.shape[0]),flags=cv2.INTER_LINEAR)    
        # binary_warped = binary_warped[200:370, 100:550]
        # print(binary_warped.shape)
        
        # cv2.imshow("result", binary_warped)
        end = time.time()
        print(1/(end-begin))
        
        #------------------------- LINE DETECTION USING PROBALISTIC HOUGH TRANSFORM ---------------------------  
        # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
        # rho : The resolution of the parameter r in pixels. We use 1 pixel.
        # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
        # threshold: The minimum number of intersections to "*detect*" a line
        # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
        # maxLineGap: The maximum gap between two points to be considered in the same line.
        # multi_lines = cv2.HoughLinesP(image=binary_warped, rho=1, theta=np.pi/180, threshold=85 , minLineLength=85,maxLineGap=20)
        # # 150 150 30 JUST FOR LONG LINE
        # line_image = display_lines(binary_warped, multi_lines)
        # combo_image = cv2.addWeighted(binary_warped, 1, line_image, 1, 1)      
        # print(combo_image)
        # cv2.imshow("final", combo_image)  
        
        #------------------------- SLIDE WINDOW METHOD ---------------------------        
        # leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        # left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        
        # out_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)    
        # cv2.imshow("slide window", out_img)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    
def imageReading():
    # VIDEO READING
    PATH_IMAGE_INPUT = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/dataset_kitty/data_road/training/image_2/"
    imageCollections = os.listdir(PATH_IMAGE_INPUT)
    PATH_IMAGE_OUTPUT = "/Users/mac/Desktop/EXTERNAL/NCKH/Autonomous-Car/report/nhan_image/roi/"
    file_object = open(PATH_IMAGE_OUTPUT + 'fps.txt', 'a')
    
    for image in imageCollections:
        begin = time.time()
        frame = cv2.imread(PATH_IMAGE_INPUT + image, cv2.IMREAD_COLOR)        
        #------------------------- PREPROCESSING IMAGE ----------------------------
        gradx = abs_sobel_thresh(frame, orient='x', thresh_min=60, thresh_max=255) # (20, 100)
        grady = abs_sobel_thresh(frame, orient='y', thresh_min=60, thresh_max=255) # (20, 100)
        c_binary = color_threshold(frame, sthresh=(100,255), vthresh=(50,255)) # GET BINARY

        preprocessImage = np.zeros_like(frame[:,:,0]) 
        preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255  # FILTER
            
        #------------------------- CREATE REGION OF INTEREST ---------------------------
        bitwiseImg = region_of_interest(preprocessImage)
        cv2.imshow("bitwise", bitwiseImg)
        cv2.imwrite(PATH_IMAGE_OUTPUT + image, bitwiseImg)
        end = time.time()
        file_object.write(str(1/(end-begin)) + "\n")
        print(1/(end-begin))
    file_object.close()
        
imageReading()
# cv2.destroyAllWindows()