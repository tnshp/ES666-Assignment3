import pdb
import glob
import cv2
import numpy as np
import os

class PanaromaStitcher:
    def __init__(self):
        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)  # Number of search iterations
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def make_panaroma_for_images_in(self, path):
       
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching.')

        images = [cv2.imread(img) for img in all_images]
        if len(images) < 2:
            raise ValueError("Need at least two images to stitch.")

        # Start with the first image as the initial panorama
        stitched_image = images[0]
        homography_matrix_list = []

        for i in range(1, len(images)):
            print(f'Stitching image {i}...')
            stitched_image, H = self.stitch_pair(stitched_image, images[i])
            homography_matrix_list.append(H)

        stitched_image = self.crop_black_borders(stitched_image)
        return stitched_image, homography_matrix_list

    def stitch_pair(self, img1, img2):
        # Detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Filter matches using Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) < 4:
            raise ValueError("Not enough good matches to compute homography.")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0, confidence=0.99)

        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        panorama_width = width1 + width2
        panorama_height = max(height1, height2)
        warped_img2 = cv2.warpPerspective(img2, H, (panorama_width, panorama_height))

       
        stitched_img = np.zeros_like(warped_img2)
        stitched_img[0:height1, 0:width1] = img1
        stitched_img = self.alpha_blend(stitched_img, warped_img2)

        return stitched_img, H

    def alpha_blend(self, img1, img2):
        """Blend two images using alpha blending to smooth transitions."""
   
        mask = (img2 > 0).astype(np.float32)  # Regions where img2 is not black
        alpha = cv2.GaussianBlur(mask, (15, 15), 0)  # Smooth mask for blending

        blended = (img1 * (1 - alpha) + img2 * alpha).astype(np.uint8)
        return blended

    def crop_black_borders(self, img):
        """Remove black regions from the stitched panorama."""
       
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

   
        return img[y:y + h, x:x + w]

    def say_hi(self):
        print("Hello! The error is fixed, and I'm working fine.")

