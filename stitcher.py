import cv2
import numpy as np

class ImageStitcher:
    def __init__(self, ransac_thresh=5.0):
        # Initialize SIFT once here, reuse it later (Optimized)
        self.sift = cv2.SIFT_create()
        self.ransac_thresh = ransac_thresh
        
    def _preprocess(self, img):
        """
        Preprocess an input image by converting it to grayscale, applying histogram 
        equalization, and median blurring.
    
        Parameters
        ----------
        img : np.ndarray
            Input image in BGR or grayscale format.
    
        Returns
        -------
        np.ndarray
            Preprocessed grayscale image.
    
        Notes
        -----
        - If the input is BGR, it is converted to grayscale.
        - Histogram equalization enhances contrast.
        - Median blur (kernel size = 3) reduces noise while preserving edges.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.medianBlur(cv2.equalizeHist(gray), 3)

    def _get_homography(self, img1, img2):
        """
        Compute the homography matrix that maps img1 onto img2 using SIFT feature 
        extraction and BFMatcher with cross-check enabled.
    
        Parameters
        ----------
        img1 : np.ndarray
            Source image.
        img2 : np.ndarray
            Destination image.
    
        Returns
        -------
        np.ndarray or None
            3x3 homography matrix if enough matches are found; otherwise None.
    
        Notes
        -----
        - Uses SIFT to detect keypoints and compute descriptors.
        - Matches descriptors with BFMatcher (crossCheck=True).
        - Requires at least 4 matches to compute homography.
        - RANSAC is used for robust homography estimation.
        """
        # Detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # Use BFMatcher to find matches
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Safety check: Need at least 4 matches for Homography
        if len(matches) < 4:
            print("Not enough matches found.")
            return None

        # Extract location of matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        return H
    def stitch(self, img1, img2, use_preprocess=False):
        """
        The main public method.
        """
        # Decide which version to use for calculation
        if use_preprocess:
            calc_img1 = self._preprocess(img1)
            calc_img2 = self._preprocess(img2)
        else:
            calc_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            calc_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        H = self._get_homography(calc_img1, calc_img2)
        
        if H is None:
            print("Could not stitch images.")
            return img2

        # Get dimensions
        h2, w2 = calc_img2.shape[:2]
        h1, w1 = calc_img1.shape[:2]

        # 1. Define Corners
        corners_img2 = np.float32([
            [0, 0], [0, h2], [w2, h2], [w2, 0]
        ]).reshape(-1, 1, 2)
        corners_img1 = np.float32([
            [0, 0], [0, h1], [w1, h1], [w1, 0]
        ]).reshape(-1, 1, 2)
    
        # 2. Warp corners of img1 to see where they land relative to img2
        warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
    
        # 3. Find the bounding box of the whole scene
        all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)
    
        [min_x, min_y] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [max_x, max_y] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
        # 4. Calculate Translation (Shift)
        offset_x = -min_x
        offset_y = -min_y
        
        dsize = (max_x - min_x, max_y - min_y)
    
        # 5. Create Translation Matrix
        M_translate = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ])
    
        # 6. Warp img1 onto the new canvas
        # We chain the shift (M) and the warp (H)
        H_final_img = M_translate @ H
        stitched_img = cv2.warpPerspective(img1, H_final_img, dsize)
    
        # 7. Paste img2 (The Anchor)
        # img2 was at (0,0). Now it is shifted by (offset_x, offset_y).
        # We simply copy-paste it into the warped canvas.
        
        # Check bounds to be safe (though logic should hold)
        y_start, y_end = offset_y, offset_y + h2
        x_start, x_end = offset_x, offset_x + w2
    
        stitched_img[y_start:y_end, x_start:x_end] = img2
        
        return stitched_image
