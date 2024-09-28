import cv2
import numpy as np

def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50

    def forward(self, img):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)
        left_lane, right_lane = self.fit_poly(img, leftx, lefty, rightx, righty)
        return out_img, left_lane, right_lane

    def pixels_in_window(self, center, margin, height):
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        self.window_height = np.int(img.shape[0] // self.nwindows)
        self.nonzero = img.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])

    def find_lane_pixels(self, img):
        histogram = hist(img)
        out_img = np.dstack((img, img, img))

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        x_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        self.extract_features(img)

        for window in range(self.nwindows):
            win_y_low = img.shape[0] - (window + 1) * self.window_height
            win_y_high = img.shape[0] - window * self.window_height

            good_left_inds = self.pixels_in_window((x_current, (win_y_low + win_y_high) // 2), self.margin, self.window_height)
            good_right_inds = self.pixels_in_window((rightx_current, (win_y_low + win_y_high) // 2), self.margin, self.window_height)

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds[0]) > self.minpix:
                x_current = np.int(np.mean(good_left_inds[0]))
            if len(good_right_inds[0]) > self.minpix:
                rightx_current = np.int(np.mean(good_right_inds[0]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = self.nonzerox[left_lane_inds]
        lefty = self.nonzeroy[left_lane_inds]
        rightx = self.nonzerox[right_lane_inds]
        righty = self.nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img, leftx, lefty, rightx, righty):
        if len(leftx) != 0 and len(rightx) != 0:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        left_lane = np.min(left_fitx)
        right_lane = np.max(right_fitx)

        for i, y in enumerate(ploty):
            cv2.line(img, (int(left_fitx[i]), int(y)), (int(left_fitx[i]), int(y)), (255, 0, 0), 10)
            cv2.line(img, (int(right_fitx[i]), int(y)), (int(right_fitx[i]), int(y)), (0, 0, 255), 10)

        return left_lane, right_lane
