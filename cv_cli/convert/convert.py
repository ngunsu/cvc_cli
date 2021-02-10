import cv2


def im2im(im_path, output_format):
    """TODO: Docstring for function.

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
    output_filename = im_path.split('.')[0] + f'.{output_format}'
    cv2.imwrite(output_filename, in_im)
