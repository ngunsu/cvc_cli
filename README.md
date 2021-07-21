# CV-CLI 

Image processing and computer vision CLI tools

---

## cvc-convert 

### im2im

Converts the image format of one or multiples images

- One image

```bash
# For example, convert img.jpg to img.png
cvc-convert im2im ./data/samples/aero1.jpg png
```

```bash
# For example, convert all the jpg images in /demo to png 
cvc-convert im2im ./data/samples/*.jpg png
```

In both cases, original images are not deleted, and the output images are created in the same folder as the original

### im2mp4

Converts a group of images into a video using **ffmpeg**. The images must have a format, e.g., frame_%d.jpg

```bash
# For example, convert the jpg images in ./data/video/frame to a mp4 video 
[poetry run] cvc-convert im2mp4 [--out_size 1920x1080] [--fps 30]  ./data/video/frame/demo%06d.jpg  output_video.mp4
```

---

## cvc-stereo

### view-images

Examples

```bash
[poetry run] cvc-stereo view-images ./data/stereo/
```

For filtering, and creating a selected list of images

```bash
[poetry run] cvc-stereo view-images ./data/stereo/ --save
```

### Stereo calibration

- Stereo calibration using chessboard 

```bash
# --help for options
[poetry run] cvc-stereo cal ./data/stereo/ 
```

The result is a mapping file stereo_params.yml

### Stereo rectification

- Using a mapping file

```bash
# --help for options
[poetry run]cvc-stereo rect ./data/stereo/ ./data/stereo/stereo_params.yml
```

Results in ./output folder

---

## cvc-mono

### Camera calibration

Example

```bash
#[poetry run] cvc-mono cal images_folder [--help for options]
cvc-mono cal ./data/stereo/left
```

The result is a mapping file calib.yml

### Camera rectification

Example

```bash
#[poetry run] cvc-mono rect images_folder calibration_file.yml [--help for options]
cvc-mono rect ./data/stereo/left ./data/calibration_files/left.yml
```

