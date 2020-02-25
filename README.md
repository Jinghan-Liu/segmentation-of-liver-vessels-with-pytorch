# segmentation-of-liver-vessels-with-pytorch

- I always forget what I have down with my codes, so that I sometimes cannot recover the code which may run well.
- This program will introduce what I have down with the CT images to do the liver vessel segmentation with pytorch.

## Image Preprocessing

- The type of the medical image is DICOM, but we can't see details of the images and use them for operations. So we need to transform them. The folder "Image Transformation" is used to transform the type and make the images visible. These codes are copied from the Internet.
  - The "imgtrans.py" is used to transform the DICOM file to mhd and raw files.
  - The "readimg.py" is used to make the mhd file visible.
