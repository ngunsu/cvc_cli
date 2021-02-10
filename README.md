# cv-cli

Image processing and computer vision CLI tools

---

## cv-convert

### im2im

Converts the image format of one or multiples images

- One image

```bash
# For example, convert img.jpg to img.png
cv-convert im2im png img.jpg
```

```bash
# For example, convert all the jpg images in /demo to png 
cv-convert im2im png /demo/*.jpg 
```

In both cases, original images are not deleted, and the output images are created in the same folder as the original

### im2mp4

Converts a group of images into a video using **ffmpeg**. The images must have a format, e.g., frame_%d.jpg

```bash
# For example, convert the jpg images in ./demo to a mp4 video 
cv-convert im2mp4 -s 1920x1080 -f 30  ./demo/frame_%d.jpg  output_video.mp4
```
