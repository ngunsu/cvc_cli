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
cvc-convert im2mp4 [--out_size 1920x1080] [--fps 30]  ./data/video/frame/demo%06d.jpg  output_video.mp4
```

