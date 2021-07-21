import cv2
import pathlib
import typer
import numpy as np
from tqdm import tqdm
from enum import Enum
from os.path import join
from os import system
from typing import Tuple
from cvc_cli.stereo.utils import get_list_of_stereo_images as get_list_of_images
from cvc_cli.stereo.calibration import stereo_calibration


class PatternType(str, Enum):
    checkerboard = 'checkerboard'


app = typer.Typer()


@app.command()
def rect(stereo_folder, cal_file, debug: bool = False, output_folder: str = './output'):
    """ Stereo rectification

    Notes:

    Spected structure of the stereo folder

    - stereo_root_path

        - left\n
            - frame_1.png\n
            - ...\n
            - frame_n.png\n
        - right\n
            - ...\n

    """
    # Reading the mapping values for stereo image rectification
    cv_file = cv2.FileStorage(cal_file, cv2.FILE_STORAGE_READ)
    left_stereo_map_x = cv_file.getNode("left_stereo_map_x").mat()
    left_stereo_map_y = cv_file.getNode("left_stereo_map_y").mat()
    right_stereo_map_x = cv_file.getNode("right_stereo_map_x").mat()
    right_stereo_map_y = cv_file.getNode("right_stereo_map_y").mat()
    cv_file.release()

    stereo_pairs, _ = get_list_of_images(stereo_folder)

    if len(stereo_pairs) > 0:
        system(f'mkdir -p {output_folder}/left')
        system(f'mkdir -p {output_folder}/right')
        system(f'mkdir -p {output_folder}/stack')

        for left_path, right_path in tqdm(stereo_pairs):
            left_im_gray = cv2.imread(left_path, 0)
            right_im_gray = cv2.imread(right_path, 0)

            # Applying stereo image rectification on the left image
            left_rect = cv2.remap(left_im_gray, left_stereo_map_x, left_stereo_map_y, cv2.INTER_LANCZOS4,
                                  cv2.BORDER_CONSTANT, 0)

            # Applying stereo image rectification on the right image
            right_rect = cv2.remap(right_im_gray, right_stereo_map_x, right_stereo_map_y, cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT, 0)

            if debug:
                stereo_stack = np.hstack((left_rect, right_rect))
                cv2.imshow('stereo_rect', stereo_stack)
                cv2.waitKey(2000)

            new_left_path = join(f'{output_folder}/left', pathlib.Path(left_path).name)
            new_right_path = join(f'{output_folder}/right', pathlib.Path(right_path).name)
            stack_path = join(f'{output_folder}/stack', pathlib.Path(right_path).name)

            stack = np.hstack((left_rect, right_rect))
            # Add lines to stack
            step = int(stack.shape[0]/10)
            for i in range(1, 10):
                pt1 = (0, step * i)
                pt2 = (stack.shape[1] - 1, step * i)
                stack = cv2.line(stack, pt1, pt2, (255, 0, 0), 1)

            cv2.imwrite(new_left_path, left_rect)
            cv2.imwrite(new_right_path, right_rect)
            cv2.imwrite(stack_path, stack)
        print('Results are in ./output')
    else:
        print('No image pair found')


@app.command()
def cal(stereo_folder, pattern_type: PatternType = PatternType.checkerboard,
        pattern_shape: Tuple[int, int] = [9, 6], pattern_size: int = 25, show: bool = False,
        debug: bool = False):
    """ Stereo calibration

    Notes:

    Spected structure of the stereo folder

    - stereo_root_path

        - left\n
            - frame_1.png\n
            - ...\n
            - frame_n.png\n
        - right\n
            - ...\n

    """
    stereo_pairs, _ = get_list_of_images(stereo_folder)

    result, left_stereo_map, right_stereo_map = stereo_calibration(stereo_pairs, pattern_type.value, pattern_shape,
                                                                   pattern_size, show)

    if result is False:
        typer.echo('Calibration pattern not found in any image', err=True)
        exit()

    params_filename = 'stereo_params.yml'
    print(f'Saving stereo mapping in {params_filename}')
    cv_file = cv2.FileStorage(params_filename, cv2.FILE_STORAGE_WRITE)
    cv_file.write('left_stereo_map_x', left_stereo_map[0])
    cv_file.write('left_stereo_map_y', left_stereo_map[1])
    cv_file.write('right_stereo_map_x', right_stereo_map[0])
    cv_file.write('right_stereo_map_y', right_stereo_map[1])
    cv_file.release()


@app.command()
def view_images(stereo_folder, save: bool = False, output_folder='./output/'):
    """Show stereo images as a horizontal stacked image (downsampled)

    Notes:

    Spected structure of the stereo folder

    - stereo_root_path

        - left\n
            - frame_1.png\n
            - ...\n
            - frame_n.png\n
        - right\n
            - ...\n
        - left_rect\n
            - ...\n
        - right_rect\n
            - ...\n
        - depth\n
            - ...\n
        - depth_npy\n
            - ...\n

    """
    stereo_pairs, subfolders = get_list_of_images(stereo_folder)

    if not save:
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
        elif key == ord('s') and save:
            frame_name = pathlib.Path(left_path).name
            for f in subfolders:
                if f in ['right', 'left', 'right_rect', 'left_rect', 'depth']:
                    system(f'cp {stereo_folder}/{f}/{frame_name} {output_folder}/{f}/')
                if f == 'depth_npy':
                    depth_name = frame_name.split('.')[0]
                    system(f'cp {stereo_folder}/{f}/{depth_name} {output_folder}/{f}/')


def main():
    app()


if __name__ == "__main__":
    main()
