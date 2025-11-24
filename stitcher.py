import cv2
import numpy as np

def preprocess_img(img):
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
    # Ensure input is essentially valid
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        image = img

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.medianBlur(image, 3)
    return image

def homography(img1, img2):
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
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Safety check: Need at least 4 matches for Homography
    if len(matches) < 4:
        print("Not enough matches found.")
        return None

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H

def stitcher(img1, img2, preprocess=False):
    """
    Warp img1 onto img2 using a computed homography, and produce a stitched 
    composite image. Handles automatic canvas expansion and translation.

    Parameters
    ----------
    img1 : np.ndarray
        The image to be warped (source image).
    img2 : np.ndarray
        The reference image (anchor).
    preprocess : bool, optional
        If True, applies preprocessing before computing homography.

    Returns
    -------
    np.ndarray
        The stitched output image containing both img1 and img2.

    Notes
    -----
    - Computes homography directly or using preprocessed images.
    - Calculates the transformed corner positions to determine required canvas size.
    - Applies translation so all warped coordinates fit in the output.
    - Img2 is paste-aligned at its translated location inside the final canvas.
    """
    # Get dimensions
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]

    # Calculate Homography (Mapping img1 -> img2)
    if preprocess:
        # Assuming you have a preprocess_img function defined elsewhere
        H = homography(preprocess_img(img1), preprocess_img(img2))
    else:
        H = homography(img1, img2)

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

    return stitched_img
