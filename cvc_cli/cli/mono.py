import typer
import cv2
import pathlib
from os import system
from os.path import join
from tqdm import tqdm
from enum import Enum
from typing import Tuple
from cvc_cli.stereo.utils import get_list_of_images
from cvc_cli.mono.calibration import mono_calibration


class PatternType(str, Enum):
    checkerboard = 'checkerboard'


app = typer.Typer()


@app.command()
def cal(image_folder, pattern_type: PatternType = PatternType.checkerboard,
        pattern_shape: Tuple[int, int] = [9, 6], pattern_size: int = 25, show: bool = False,
        debug: bool = False, output_filename='calib.yml'):
    """ Monocular calibration

    Example:

    cvc-mono cal ./data/stereo/left
    """

    paths = get_list_of_images(image_folder)

    result, mtx, dist, rvecs, tvecs, w, h = mono_calibration(paths, pattern_type.value, pattern_shape,
                                                             pattern_size, show)

    if result is False:
        typer.echo('Calibration pattern not found in any image', err=True)
        exit()

    # Before saving the files, convert the intrinsics to deal with black pixels. Create a scaled image
    # without missed pixels
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    print(f'Saving camera mapping in {output_filename}')
    cv_file = cv2.FileStorage(output_filename, cv2.FILE_STORAGE_WRITE)
    cv_file.write('roi', roi)
    cv_file.write('intrinsics', mtx)
    cv_file.write('dist', dist)
    cv_file.write('mapx', mapx)
    cv_file.write('mapy', mapy)
    cv_file.release()


@app.command()
def rect(image_folder, cal_file, debug: bool = False, use_roi: bool = True, output_folder: str = './output'):
    """ Rectify images from a given folder using a cal_file

    Example:

    cvc-mono rect ./data/stereo/left ./data/calibration_files/left.yml
    """
    cv_file = cv2.FileStorage(cal_file, cv2.FILE_STORAGE_READ)
    roi = cv_file.getNode("roi").mat()
    mapx = cv_file.getNode("mapx").mat()
    mapy = cv_file.getNode("mapy").mat()
    cv_file.release()

    paths = get_list_of_images(image_folder)

    if len(paths) > 0:
        system(f'mkdir -p {output_folder}')

        for im_path in tqdm(paths):
            im_gray = cv2.imread(im_path, 0)

            # Applying stereo image rectification on the left image
            im_rect = cv2.remap(im_gray, mapx, mapy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            x, y, w, h = roi
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            im_rect = im_rect[y:y+h, x:x+w]
            new_path = join(f'{output_folder}', pathlib.Path(im_path).name)
            cv2.imwrite(new_path, im_rect)

    print(f'Rectification ended, results are inside {output_folder}')


def main():
    app()


if __name__ == "__main__":
    main()
