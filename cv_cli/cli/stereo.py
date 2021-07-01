import click
import cv2
import os
import pathlib
import numpy as np
from glob import glob
from tqdm import tqdm
from os import system


def get_list_of_images(stereo_root_path):
    """ Reads stereo folder and returns a list of stereo images pairs.

    Notes
    -----

    Stereo root path, must have the following structure

    - stereo_root_path
        - left
            - frame_1.png
            - ...
            - frame_n.png
        - right
            - frame_1.png
            - ...
            - frame_n.png
        - left_rect
            - frame_1.png
            - ...
            - frame_n.png
        - right_rect
            - frame_1.png
            - ...
            - frame_n.png
        - depth
            - frame_1.png
            - ...
            - frame_n.png
        - depth_npy
            - frame_1.npy
            - ...
            - frame_n.npy

    Parameters
    ----------
    stereo_root_path: string
        Root path of the stereo dataset.

    Returns
    -------
    list, list
        List of stero pairs paths, and subfolders (folder structure)

    """
    # Read folder structure
    path = pathlib.Path(stereo_root_path)
    subfolders = [x.name for x in path.iterdir() if x.is_dir()]
    minimun_folders = ['left', 'right']
    if not set(minimun_folders).issubset(set(subfolders)):
        exit('The folders doesnt contain the minimum required folders {subfolders }(check documentation)')

    # Read stereo pairs
    stereo_pairs = []
    left_folder = os.path.join(stereo_root_path, 'left')
    left_list = glob(f'{left_folder}/frame_*.*')
    left_list = sorted(left_list)
    for left_im_path in left_list:
        name = pathlib.Path(left_im_path).name
        right_im_path = os.path.join(stereo_root_path, 'right', name)
        if pathlib.Path(right_im_path).exists():
            stereo_pairs.append((left_im_path, right_im_path))
        else:
            print(f'Missing pair {right_im_path}')
    return stereo_pairs, subfolders


@click.group()
def cli():
    pass


@cli.command()
@click.option('-c', '--cal_file', type=click.Path(exists=True), help='Calibration file')
@click.option('-o', '--output_folder', type=str, default='output_rect', help='Output path to store saved images')
@click.option('--save/--no-save', type=bool, default=False, help='Store rectified images')
@click.option('--stepbystep/--no-stepbystep', type=bool, default=True,
              help='Require input from keyboard to continue after one image')
@click.argument('stereo_root_path')
def stereo_rect(cal_file, output_folder, save, stepbystep, stereo_root_path):
    # @TODO: Finish the implementation of this method
    pass


@cli.command()
@click.option('-p', '--pattern_type', help='Pattern type', default='chessboard', type=click.Choice(['chessboard']))
@click.option('-psh', '--pattern_shape', nargs=2, type=(int, int), default=(9, 7), help='Pattern size (e.g., 9 6)')
@click.option('-psi', '--pattern_size', type=int, default=1, help='Pattern element size in mm')
@click.option('-o', '--output_folder', type=str, default='output_cal', help='Output path if clean')
@click.option('--show/--no-show', type=bool, default=True, help='Show chessboard detection results')
@click.argument('stereo_root_path')
def cal_stereo(stereo_root_path, pattern_type, pattern_shape, pattern_size, output_folder, show):
    stereo_pairs, _ = get_list_of_images(stereo_root_path)

    # Termination criteria for refining the detected corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, pattern_shape[0] * pattern_shape[1], 3), np.float32)
    # This represents the real dimensions of the chess board squares
    objp[0, :, :2] = np.mgrid[0:pattern_shape[0], 0:pattern_shape[1]].T.reshape(-1, 2)
    objp *= pattern_size

    # Prepare image point containers
    img_ptsL = []
    img_ptsR = []
    obj_pts = []

    # To retain image informatin after the loop
    left_im_gray = None
    right_im_gray = None

    # Stadistics
    total = len(stereo_pairs)
    used = 0
    for left_path, right_path in tqdm(stereo_pairs):
        left_im = cv2.imread(left_path)
        right_im = cv2.imread(right_path)

        left_im_gray = cv2.imread(left_path, 0)
        right_im_gray = cv2.imread(right_path, 0)

        if pattern_type == 'chessboard':
            exists_pattern_left, corners_left = cv2.findChessboardCorners(left_im_gray, pattern_shape, None)
            exists_pattern_right, corners_right = cv2.findChessboardCorners(right_im_gray, pattern_shape, None)

            if exists_pattern_left and exists_pattern_right:
                used += 2
                obj_pts.append(objp)
                cv2.cornerSubPix(right_im_gray, corners_right, (11, 11), (-1, -1), criteria)
                cv2.cornerSubPix(left_im_gray, corners_left, (11, 11), (-1, -1), criteria)

                left_im = cv2.drawChessboardCorners(left_im, pattern_shape, corners_left, exists_pattern_left)
                right_im = cv2.drawChessboardCorners(right_im, pattern_shape, corners_right, exists_pattern_right)
                stereo_pair = np.hstack((left_im, right_im))
                if show:
                    cv2.imshow('stero_pair', stereo_pair)
                    cv2.waitKey(30)

                img_ptsL.append(corners_left)
                img_ptsR.append(corners_right)
            else:
                print(f'Pattern was not detected on {left_path},{right_path}')
                print(f'left pattern {exists_pattern_left}, right pattern {exists_pattern_right}')

    # Calibrating left camera
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, left_im_gray.shape[::-1], None, None)
    hL, wL = left_im_gray.shape[:2]
    new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

    # Calibrating right camera
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, right_im_gray.shape[::-1], None, None)
    hR, wR = right_im_gray.shape[:2]
    new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

    # Stereo camera calibration
    flags = 1
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR,
                                                                                        new_mtxL, distL, new_mtxR,
                                                                                        distR, left_im_gray.shape[::-1],
                                                                                        criteria_stereo, flags)
    print('baseline', Trns[0], 'mm')
    print('Intrinsic left camera', new_mtxL)
    print('Intrinsic right camera', new_mtxR)

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

    print("Saving stereo mappings......")
    system(f'mkdir -p {output_folder}')
    cv_file = cv2.FileStorage(f'{output_folder}/stereo_params.xml', cv2.FILE_STORAGE_WRITE)
    cv_file.write('left_stereo_map_x', Left_Stereo_Map[0])
    cv_file.write('left_stereo_map_y', Left_Stereo_Map[1])
    cv_file.write('right_stereo_map_x', Right_Stereo_Map[0])
    cv_file.write('right_stereo_map_y', Right_Stereo_Map[1])
    cv_file.release()

    print(f' From a total of {total} image pairs, {used} were used')


@cli.command()
@click.option('--clean/--noclean', type=bool, default=False, help='Save selected images pairs in an output_folder')
@click.option('-o', '--output_folder', type=str, default='output', help='Output path if clean')
@click.argument('stereo_root_path')
def view_images(stereo_root_path, clean, output_folder):
    """Show stereo images as a horizontal stacked image (downsampled)
    """
    stereo_pairs, subfolders = get_list_of_images(stereo_root_path)

    if not clean:
        print('Commands  n: next, q:quit')
    else:
        system(f'mkdir -p {output_folder}')
        for f in subfolders:
            system(f'mkdir -p {output_folder}/{f}')
        print('Commands  n: next, q:quit, s:save')

    for left_path, right_path in stereo_pairs:
        left_im = cv2.imread(left_path)
        right_im = cv2.imread(right_path)

        stereo_im = np.hstack((left_im, right_im))
        cv2.imshow('stereo_pair', stereo_im)
        key = cv2.waitKey(0)
        if key == ord('q'):
            quit()
        elif key == ord('n'):
            continue
        elif key == ord('s'):
            frame_name = pathlib.Path(left_path).name
            for f in subfolders:
                if f in ['right', 'left', 'right_rect', 'left_rect', 'depth']:
                    system(f'cp {stereo_root_path}/{f}/{frame_name} {output_folder}/{f}/')
                if f == 'depth_npy':
                    depth_name = frame_name.split('.')[0]
                    system(f'cp {stereo_root_path}/{f}/{depth_name} {output_folder}/{f}/')


if __name__ == "__main__":
    cli()
