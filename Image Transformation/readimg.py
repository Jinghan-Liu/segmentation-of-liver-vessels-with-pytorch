import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

img_path = './SaveRaw/sample.mhd'


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    return numpyImage


numpyImage = load_itk_image(img_path)
# 此mhd文件中切片的数量
slice = 74


for i in range(slice):
    image = np.squeeze(numpyImage[i])
    # 以灰度图的形式显示出图片
    plt.imshow(image, cmap='gray')
    plt.show()
