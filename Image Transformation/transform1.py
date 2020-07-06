import cv2
import os
import pydicom
import numpy
import SimpleITK

# numpy.set_printoptions(threshold=numpy.inf)

PathDicom='./LABELLED_DICOM/'
SaveRawDicom='./SaveRaw/'
lstFileDCM=[]

for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        # lstFileDCM就是每一张图片的路径，如：'./LABELLED_DICOM/image_0'
        lstFileDCM.append(os.path.join(dirName, filename))


# 读取第一张dicom图片
RefDs = pydicom.read_file(lstFileDCM[0])
# RefDs中包含了这张dicom图片中的所有信息，很多很多......
# 可以通过pixel_array获取像素数据


# 得到dicom图片所组成3D图片的维度
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFileDCM))
# (512,512,74)
# 74代表的是这里有74张dicom图片
# 希望的是通过图片预处理，能够将每一个病人的lstFileDCM的len变为一个固定的，如64，以便进行卷积


# 得到x方向和y方向的Spacing并得到z方向的层厚
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),  float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
# (0.742187976837158, 0.742187976837158, 2.5)
# PixelSpacing - 每个像素点实际的长度与宽度,单位(mm)
# SliceThickness - 每层切片的厚度,单位(mm)


# 得到图像的原点
Origin = RefDs.ImagePositionPatient
# [0, 0, 0]
# 这里也就是图片左下角的点的坐标


# 根据维度创建一个numpy的三维数组，并将元素类型设为：pixel_array.dtype，当前元素全为0
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)


# 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
i = 0
for filenameDCM in lstFileDCM:
    ds = pydicom.read_file(filenameDCM)
    ArrayDicom[:, :, lstFileDCM.index(filenameDCM)] = ds.pixel_array
    cv2.imwrite("./image/patient/out_"+str(i)+'.png', ArrayDicom[:, :, lstFileDCM.index(filenameDCM)])
    i += 1


# 对numpy数组进行转置，即把坐标轴(x,y,z)变换为(z,y,x)，这样是dicom存储文件的格式，即第一个维度为z轴便于图片堆叠
ArrayDicom = numpy.transpose(ArrayDicom, (2, 0, 1))
img_path = './image/label/'
slice = 74


# # 在变成.mhd文件之前，进行直方图均衡化操作
# for i in range(slice):
#     img = cv2.imread(img_path+"out_"+str(i)+".png", 0)
#     print(i)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     dst = clahe.apply(img)
#     for k in range(512):
#         for j in range(512):
#             ArrayDicom[i][k][j] = dst[k][j]
#
#
# # 将现在的numpy数组通过SimpleITK转换为mhd和raw文件
# sitk_img = SimpleITK.GetImageFromArray(ArrayDicom, isVector=False)
# sitk_img.SetSpacing(ConstPixelSpacing)
# sitk_img.SetOrigin(Origin)
# SimpleITK.WriteImage(sitk_img, os.path.join(SaveRawDicom, "example_segment" + ".mhd"))
# # 如果仅仅是通过上面的这几个量就可以生成.mhd文件的话
# # 那么其实对于.jpg，.png图片也可以，只需要自己手动设置一下这些量
# # 自己设定规则自己重新构建一个三维的像素数组，然后转换为.mhd文件

# 下面是一次尝试
tr_img=SimpleITK.GetImageFromArray(ArrayDicom,isVector=False)
tr_img.SetSpacing(ConstPixelSpacing)
tr_img.SetOrigin(Origin)
SimpleITK.WriteImage(tr_img,os.path.join(SaveRawDicom,"try_image"+".mhd"))
# 实践证明，任意一个三维的numpy数组都可以转换为一个.mhd文件
# Origin可以设置为原点
# 就是ConstPixelSpacing可能不一样，不知道对Vnet有无影响
