import cv2
import numpy as np
from tqdm import tqdm


def compute_error(obj_pts, img_pts, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(obj_pts)):
        imgpoints2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f'total error: {mean_error/len(obj_pts)}')


def stereo_calibration(stereo_pairs, pattern_type, pattern_shape, pattern_size, show, debug=False):
    """ Calibrates a stereo camera

    Parameters
    ----------
    stereo_pairs (list):
        list of stereo images path
    pattern_type (str):
        Calibration pattern type
    pattern_size (list):
        Number of valid squares, for example (9,7)
    show (bool):
        Show the calibration process (patterns and detections)
    debug (bool):
        If debug, shows more info

    Returns
    -------
    left_map, right_map
        Mappings for stereo rectification
    """

    # Termination criteria for refining the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, pattern_shape[0] * pattern_shape[1], 3), np.float32)
    # This represents the real dimensions of the chess board squares
    objp[0, :, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2)
    objp *= pattern_size/1000

    # Prepare image point containers
    img_ptsL = []
    img_ptsR = []
    obj_pts = []

    # To retain image information after the loop
    left_im_gray = None
    right_im_gray = None

    # Statistics
    total = len(stereo_pairs)
    used = 0
    for left_path, right_path in tqdm(stereo_pairs):
        left_im = cv2.imread(left_path)
        right_im = cv2.imread(right_path)

        left_im_gray = cv2.imread(left_path, 0)
        right_im_gray = cv2.imread(right_path, 0)

        if pattern_type == 'checkerboard':
            exists_pattern_left, corners_left = cv2.findChessboardCorners(left_im_gray, pattern_shape, None)
            exists_pattern_right, corners_right = cv2.findChessboardCorners(right_im_gray, pattern_shape, None)

            if exists_pattern_left and exists_pattern_right:
                used += 1
                obj_pts.append(objp)
                cv2.cornerSubPix(right_im_gray, corners_right, (11, 11), (-1, -1), criteria)
                cv2.cornerSubPix(left_im_gray, corners_left, (11, 11), (-1, -1), criteria)

                left_im = cv2.drawChessboardCorners(left_im, pattern_shape, corners_left, exists_pattern_left)
                right_im = cv2.drawChessboardCorners(right_im, pattern_shape, corners_right, exists_pattern_right)
                stereo_pair = np.hstack((left_im, right_im))
                if show:
                    cv2.imshow('stereo_pair', stereo_pair)
                    cv2.waitKey(1000)

                img_ptsL.append(corners_left)
                img_ptsR.append(corners_right)
            else:
                if debug:
                    print(f'Pattern was not detected on {left_path},{right_path}')
                    print(f'left pattern {exists_pattern_left}, right pattern {exists_pattern_right}')
    if used == 0:
        return False, False, False
    # Calibrating left and right camera
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, left_im_gray.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, right_im_gray.shape[::-1], None, None)

    print('Intrinsic left camera', mtxL)
    compute_error(obj_pts, img_ptsL, rvecsL, tvecsL, mtxL, distL)
    print('Intrinsic right camera', mtxR)
    compute_error(obj_pts, img_ptsR, rvecsR, tvecsR, mtxR, distR)

    # hL, wL = left_im_gray.shape[:2]
    # new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))
    new_mtxL = mtxL

    # Calibrating right camera
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, right_im_gray.shape[::-1], None, None)
    # hR, wR = right_im_gray.shape[:2]
    # new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))
    new_mtxR = mtxR

    # Stereo camera calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camera matrices so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and fundamental matrix
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR,
                                                                                        new_mtxL, distL, new_mtxR,
                                                                                        distR, left_im_gray.shape[::-1],
                                                                                        criteria_stereo, flags)
    print('baseline', Trns[0]*1000, 'mm')

    # Stereo rectification
    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                                                              left_im_gray.shape[::-1], Rot, Trns,
                                                                              rectify_scale, (0, 0))
    # Compute and saving the mappings
    Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l,
                                                  left_im_gray.shape[::-1], cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r,
                                                   right_im_gray.shape[::-1], cv2.CV_16SC2)
    print(f' From a total of {total} image pairs, {used} were used')
    return True, Left_Stereo_Map, Right_Stereo_Map
