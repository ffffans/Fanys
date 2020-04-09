from unet import *
from image_util import *


output_path = "D:\\model\\unet_fan\\output\\"
data_povider = ImageDataProvider("D:\\model\\train\\*.tif")

net = UnetModel(img_width=64, img_height=64, img_channel=3)
trainer = Trainer(net)
path = trainer.train(data_povider, output_path, iteration=50, epochs=10)
