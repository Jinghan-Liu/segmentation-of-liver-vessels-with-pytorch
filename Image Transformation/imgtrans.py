import cv2
import os
import pydicom
import numpy
import SimpleITK

PathDicom='./LABELLED_DICOM/'    # 存储原始DICOM文件的位置
SaveRawDicom='./SaveRaw/'        # 存储最终转换后的mhd+raw文件的位置
lstFileDCM=[]

for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        lstFileDCM.append(os.path.join(dirName, filename))


# 读取第一张dicom图片
RefDs = pydicom.read_file(lstFileDCM[0])
# 得到dicom图片所组成3D图片的维度
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFileDCM))
# (512,512,74)【这是我的一张示例图片输出的结果】

# 得到x方向和y方向的Spacing并得到z方向的层厚
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),  float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
# (0.742187976837158, 0.742187976837158, 2.5)【这是我的一张示例图片输出的结果】

# 得到图像的原点
Origin = RefDs.ImagePositionPatient
# [0, 0, 0]【这是我的一张示例图片输出的结果】

# 根据维度创建一个numpy的三维数组，并将元素类型设为：pixel_array.dtype
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
i = 0
for filenameDCM in lstFileDCM:
    ds = pydicom.read_file(filenameDCM)
    ArrayDicom[:, :, lstFileDCM.index(filenameDCM)] = ds.pixel_array
    # 将文件按照png的格式写进当前目录
    cv2.imwrite("out_"+str(i)+'.png', ArrayDicom[:, :, lstFileDCM.index(filenameDCM)])
    i += 1


# 对numpy数组进行转置，即把坐标轴(x,y,z)变换为(z,y,x)，这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
ArrayDicom = numpy.transpose(ArrayDicom, (2, 0, 1))

# 将现在的numpy数组通过SimpleITK转换为mhd和raw文件
sitk_img = SimpleITK.GetImageFromArray(ArrayDicom, isVector=False)
sitk_img.SetSpacing(ConstPixelSpacing)
sitk_img.SetOrigin(Origin)
SimpleITK.WriteImage(sitk_img, os.path.join(SaveRawDicom, "sample" + ".mhd"))
# 虽然最后只写了是sample.mhd的文件，但实际上在同一个文件夹中还会生成sample.raw的文件
