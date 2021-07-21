import typer
from typing import List
from os import system
from joblib import Parallel, delayed
from cvc_cli.convert import convert


app = typer.Typer()


@app.command()
def im2im(src: List[str], output_format, jobs=-1):
    """ Parallel conversion of  multiples image to a given output_format

    Example:

        pim2im src.jpg png [jobs=-1]

        pim2im data/sample/*.png jpg --jobs 8
    """
    print(src)
    if len(src) == 1:
        convert.im2im(src[0], output_format)
    else:
        Parallel(n_jobs=jobs)(delayed(convert.im2im)(path, output_format) for path in src)


@app.command()
def im2mp4(src_template, output_filename, fps=30, out_size='1920x1080'):
    """ Convert images to video using ffmpeg

    Example:

        im2mp4 data/video_images/frame_%d.png out.mp4 [--fps 30] [--out_size 1920x1080]
    """
    system(f'ffmpeg -r {fps} -f image2 -s {out_size} -i {src_template} -vcodec libx264 -crf 25 {output_filename}')


def main():
    app()


if __name__ == "__main__":
    main()
