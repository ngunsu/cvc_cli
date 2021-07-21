import cv2
import pathlib


def im2im(im_path, output_format):
    """Convert one image format to output format

    Parameters
    ----------
    input_filename (str): Image path
    output_format (str): Output image format

    Returns
    -------
    None
    """
    in_im = cv2.imread(im_path)
    if in_im.shape[2] == 1 and output_format != 'pgm':
        in_im = cv2.cvtColor(in_im, cv2.COLOR_GRAY2BGR)
    if in_im.shape[2] >= 1 and output_format == 'pgm':
        in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2GRAY)
    im_path = pathlib.Path(im_path)
    output_filename = im_path.name.replace(im_path.suffix, f'.{output_format}')
    cv2.imwrite(output_filename, in_im)
