from PIL import Image
import SimpleITK
import numpy
import os

path = './train_org/label/patient1/'

constpixeldims = (256, 256, 256)
images = numpy.zeros(constpixeldims)

for i in range(256):
    img_path = path + 'patient_1_' + str(i) + '.png'
    img = Image.open(img_path)
    images[i] = img

constpixelspacing = (0.742187976837158, 0.742187976837158, 2.5)

origin = [0, 0, 0]

tr_img = SimpleITK.GetImageFromArray(images, isVector=False)
tr_img.SetSpacing(constpixelspacing)
tr_img.SetOrigin(origin)
SimpleITK.WriteImage(tr_img, os.path.join(path, "patient_1" + ".mhd"))
