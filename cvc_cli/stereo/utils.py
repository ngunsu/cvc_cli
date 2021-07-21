import pathlib
import os
from glob import glob


def get_list_of_stereo_images(stereo_folder):
    """ Reads stereo folder and returns a list of stereo images pairs.

    Notes
    -----

    root path, must have the following structure

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
    stereo_folder: string
        Root path of the stereo dataset.

    Returns
    -------
    list, list
        List of stero pairs paths, and subfolders (folder structure)

    """
    # Read folder structure
    path = pathlib.Path(stereo_folder)
    subfolders = [x.name for x in path.iterdir() if x.is_dir()]
    minimun_folders = ['left', 'right']
    if not set(minimun_folders).issubset(set(subfolders)):
        exit('The folders doesnt contain the minimum required folders {subfolders }(check documentation)')

    # Read stereo pairs
    stereo_pairs = []
    left_folder = os.path.join(stereo_folder, 'left')
    image_types = ['.png', '.jpg', 'jpeg', 'tiff', 'bmp']
    left_list = [im_path for im_path in glob(f'{left_folder}/*.*') if pathlib.Path(im_path).suffix in image_types]
    left_list = sorted(left_list)
    for left_im_path in left_list:
        name = pathlib.Path(left_im_path).name
        right_im_path = os.path.join(stereo_folder, 'right', name)
        if pathlib.Path(right_im_path).exists():
            stereo_pairs.append((left_im_path, right_im_path))
        else:
            print(f'Missing pair {right_im_path}')
    return stereo_pairs, subfolders


def get_list_of_images(image_folder):
    """ Reads images in a folder and returns a list of images paths.

    Parameters
    ----------
    image_folder: string
        Folder that contains images.

    Returns
    -------
    list
        List of images path

    """
    # Read folder
    path = pathlib.Path(image_folder)
    image_types = ['.png', '.jpg', 'jpeg', 'tiff', 'bmp']
    paths = [str(x.absolute()) for x in path.iterdir() if x.suffix in image_types]
    return paths
