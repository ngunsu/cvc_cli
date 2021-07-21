import cv2
import numpy as np
from tqdm import tqdm


def mono_calibration(images, pattern_type, pattern_shape, pattern_size, show=False, debug=False):
    """ Calibrates a stereo camera

    Parameters
    ----------
    images(list):
        list of images paths
    pattern_type (str):
        Calibration pattern type
    pattern_shape(list):
        Number of valid squares, for example (9,7)
    pattern_size(int):
        Pattern size in mm
    show (bool):
        Show the calibration process (patterns and detections)
    debug (bool):
        If debug, shows more info

    Returns
    -------
    result, intrinsic, distortions, etc.
       Calibration parameters
    """

    # Termination criteria for refining the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, pattern_shape[0] * pattern_shape[1], 3), np.float32)

    # This represents the real dimensions of the chess board squares
    objp[0, :, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2)
    objp *= pattern_size/1000
    print(pattern_size/1000)

    # Prepare image point containers
    img_pts = []
    obj_pts = []

    # To retain image informatin after the loop
    im_gray = None

    # Stadistics
    total = len(images)
    used = 0
    for image_path in tqdm(images):
        im = cv2.imread(image_path)
        im_gray = cv2.imread(image_path, 0)

        if pattern_type == 'checkerboard':
            exists_pattern, corners = cv2.findChessboardCorners(im_gray, pattern_shape, None)

            if exists_pattern:
                used += 1
                obj_pts.append(objp)
                cv2.cornerSubPix(im_gray, corners, (11, 11), (-1, -1), criteria)

                im = cv2.drawChessboardCorners(im, pattern_shape, corners, exists_pattern)
                if show:
                    cv2.imshow('image', im)
                    cv2.waitKey(1000)

                img_pts.append(corners)
            else:
                if debug:
                    print(f'Pattern was not detected on {image_path}')
    if used == 0:
        return False, None, None, None, None, None, None
    else:
        print(f'used {used}/{total}')
    # Calibrating camera
    flags = 0
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, im_gray.shape[::-1], None, None, flags=flags)
    print('Intrinsic matrix')
    print(mtx)
    print('Distorsion parameters')
    print(dist)
    mean_error = 0
    for i in range(len(obj_pts)):
        imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f'total error: {mean_error/len(obj_pts)}')

    return ret, mtx, dist, rvecs, tvecs, im_gray.shape[1], im_gray.shape[0]
