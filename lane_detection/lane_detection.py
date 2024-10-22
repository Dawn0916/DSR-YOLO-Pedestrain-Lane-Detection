import numpy as np
import cv2
import time
from threading import Thread
from queue import Queue
import os
import subprocess
import moviepy.editor as moviepy

# defualt video number, if you want to process the "fog_video.mp4", change video_index to 1
#video_index = 1

# the result of lane detection, we add the road to the main frame
# road = np.zeros((720, 1280, 3))

# A flag which means the process is started
#started = 0

# Pipeline combining color and gradient thresholding
def thresholding_pipeline(img, s_thresh=(90, 255), sxy_thresh=(20, 100)):

    img = np.copy(img)
    # 1: Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float32)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # 2: Calculate x directional gradient
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sxy_thresh[0]) & (scaled_sobelx <= sxy_thresh[1])] = 1
    grad_thresh = sxbinary

    # 3: Color Threshold of s channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # 4: Combine the two binary thresholds
    combined_binary = np.zeros_like(grad_thresh)
    combined_binary[(s_binary == 1) | (grad_thresh == 1)] = 1

    return combined_binary

# Apply perspective transformation to bird's eye view
def perspective_transform(img, src_mask, dst_mask):

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_mask)
    dst = np.float32(dst_mask)
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_img


# Implement Sliding Windows and Fit a Polynomial
def sliding_windows(binary_warped, nwindows=9):

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int32(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
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
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, lefty, leftx, righty, rightx


# Warp lane line projection back to original image
def project_lanelines(binary_warped, orig_img, left_fit, right_fit, dst_mask, src_mask):

    #global road
    # global started
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warped_inv = perspective_transform(color_warp, dst_mask, src_mask)
    road = warped_inv
    started = 1
    # print(f"pts_left: {pts_left}")
    # print(f"pts_right: {pts_right}")
    # print(f"pts: {pts}")
    # print(f"road: {road}")
    return road, started

# Main process functions
def main_pipeline(input):

    # step 1 select the ROI, and we need to distort the image for fog_video
    # if video_index == 0:
    #     image = input
    #     top_left = [540, 460]
    #     top_right = [754, 460]
    #     bottom_right = [1190, 670]
    #     bottom_left = [160, 670]
    # else:
    mtx = np.array([[1.15396467e+03, 0.00000000e+00, 6.69708251e+02],[0.00000000e+00, 1.14802823e+03, 3.85661017e+02], 
                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[-2.41026561e-01, -5.30262184e-02, -1.15775369e-03, -1.27924043e-04, 2.66417032e-02]])
    image = cv2.undistort(input, mtx, dist, None, mtx)
    # top_left = [240, 270]
    # top_right = [385, 270]
    # bottom_right = [685, 402]
    # bottom_left = [0, 402]
    top_left = [540, 460]
    top_right = [754, 460]
    bottom_right = [1190, 670]
    bottom_left = [160, 670]
    src_mask = np.array([[(top_left[0], top_left[1]), (top_right[0], top_right[1]),
                          (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])]], np.int32)
    dst_mask = np.array([[(bottom_left[0], 0), (bottom_right[0], 0),
                          (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])]], np.int32)

    # Step 2 Thresholding: color and gradient thresholds to generate a binary image
    binary_image = thresholding_pipeline(image, s_thresh=(90, 255))

    # Step 3 Perspective transform on binary image:
    binary_warped = perspective_transform(binary_image, src_mask, dst_mask)

    # Step 4 Fit Polynomial
    left_fit, right_fit, lefty, leftx, righty, rightx = sliding_windows(binary_warped, nwindows=9)

    # # Debugging
    # print(f"left_fit: {left_fit}")
    # print(f"right_fit: {right_fit}")
    # print(f"lefty: {lefty}")
    # print(f"leftx: {leftx}")
    # print(f"righty: {righty}")
    # print(f"rightx: {rightx}")

    # print(f"len_left_fit: {len(left_fit)}")
    # print(f"len_right_fit: {len(right_fit)}")
    # print(f"len_lefty: {type(lefty)}")
    # print(f"len_leftx: {type(leftx)}")
    # print(f"len_righty: {type(righty)}")
    # print(f"len_rightx: {type(rightx)}")
    # print(f"len_lefty: {len(lefty)}")
    # print(f"len_leftx: {len(leftx)}")
    # print(f"len_righty: {len(righty)}")
    # print(f"len_rightx: {len(rightx)}")



    # Step 5 Project Lines
    road, started = project_lanelines(binary_warped, image, left_fit, right_fit, dst_mask, src_mask)
    
    return road, started, leftx, lefty, rightx, righty

if __name__ == '__main__':
    outputdir=os.getcwd()
    path = os.path.join(outputdir, "data", "road_0160" + ".mp4")
    frames_counts = 1
    cap=cv2.VideoCapture(path) 

    # Step: Save the video in AVI format using XVID codec
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
    # fps = 20.0  # Frames per second
    # frame_width = int(cap.get(3))  # Get frame width
    # frame_height = int(cap.get(4))  # Get frame height
    # output_video_path_avi = 'test/lane_detection_output_video.avi'
    # out = cv2.VideoWriter(output_video_path_avi, fourcc, fps, (frame_width, frame_height))    

    # Step: Save the video in MP4 format using mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files
    fps = 20.0  # Frames per second
    frame_width = int(cap.get(3))  # Get frame width
    frame_height = int(cap.get(4))  # Get frame height
    output_video_path_avi = 'lane_detection/road_0160_lane_detection_output_video.mp4'

    out = cv2.VideoWriter(output_video_path_avi, fourcc, fps, (frame_width, frame_height))    

    class MyThread(Thread):
        def __init__(self, q):
            Thread.__init__(self)
            self.q = q
            self.road = np.zeros((720, 1280, 3))  # Initialize 'road' as None to hold the result later
            self.started = 0
            self.leftx = np.empty(7988)
            self.lefty = np.empty(7988)
            self.rightx = np.empty( 20229)
            self.righty = np.empty( 20229)

        def run(self):
            while(1):
                if (not self.q.empty()):
                    image = self.q.get()
                    self.road, self.started, self.leftx, self.lefty, self.rightx, self.righty = main_pipeline(image)
                    lane_left_coordinates = np.column_stack((self.leftx, self.lefty))
                    lane_right_coordinates = np.column_stack((self.rightx, self.righty))

                    np.savetxt("lane_detection/road_0160_lane_left_coordinates.csv", lane_left_coordinates, delimiter=",", fmt="%.2f")
                    np.savetxt("lane_detection/road_0160_lane_right_coordinates.csv", lane_right_coordinates, delimiter=",", fmt="%.2f")
                    break  # Stop the loop after processing one item (if needed)
        
        def get_road(self):
            return self.road, self.started, self.leftx, self.lefty, self.rightx, self.righty

    q = Queue()
    q.queue.clear()
    thd1 = MyThread(q)
    # thd1.setDaemon(True)
    thd1.daemon = True
    # Access the 'road' variable from the thread

    thd1.start()

    while (True):  
            start=time.time()
            ret,frame=cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            if frame is None:
                print("Error: Frame is None. Check if the video capture is working.")
                break
            # Detect the lane every 5 frames
            if frames_counts % 5 == 0:
                q.put(frame)
            # Add the lane image on the original frame if started
            # road,started = thd1.get_road()
            road,started,_,_,_,_  = thd1.get_road()
            
            if road is None:
                print("Error: Road is None. Check the image processing pipeline.")
                break

            if started:
                # Ensure 'road' is the same size as 'frame'
                if frame.shape[:2] != road.shape[:2]:
                    road = cv2.resize(road, (frame.shape[1], frame.shape[0]))

                # Ensure 'road' has the same number of channels as 'frame'
                if len(road.shape) == 2:  # If 'road' is grayscale
                    road = cv2.cvtColor(road, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR

                # Ensure both 'frame' and 'road' have the same data type
                if road.dtype != frame.dtype:
                    road = road.astype(frame.dtype)

                # Blend the images using cv2.addWeighted
                frame = cv2.addWeighted(frame, 1.0, road, 0.5, 0.0)
                #frame = cv2.addWeighted(frame, 1, road, 0.5, 0.0)
                
            # Write the processed frame to the AVI video file
            out.write(frame)
            cv2.imshow("Real-time lane detection", frame)

                
            if cv2.waitKey(1)&0xFF==ord('q'):  
                break  
            
            frames_counts+=1
            cv2.waitKey(12)
            finish=time.time()
            # print('FPS:  ' + str(int(1/(finish-start)))) 

    cap.release()  
    cv2.destroyAllWindows() 
