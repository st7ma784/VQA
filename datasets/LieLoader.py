
from sklearn.preprocessing import normalize
import numpy as np
import torch

from PIL import Image
import sys
from pathlib import Path

import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import pandas as pd

csvlocation="../lieDetection"
imgfolder=os.path.join(csvlocation,"images")
responses=pd.read_csv(os.path.join(csvlocation,'responses.csv'), index_col=0 )
imgnames=pd.read_csv(os.path.join(csvlocation,'image file names.csv'), index_col=0 )
picture_orders=pd.read_csv(os.path.join(csvlocation,'picture_orders.csv'), index_col=0 )
## Let's check its what we think should be there
print("Read in PictureOrders: {}".format(picture_orders.columns))
print("Read in ImageNames: {}".format(imgnames.columns))
print("Read in Responses: {}".format(responses.columns))
#Some house keeping for useful formats
Participants=picture_orders.to_dict('records') # unique in   # [no.: person, dis_orders_Des: imageorders] 
files=imgnames.to_dict('list')
# some images may be misssing so lets check...
#print([i for i in files['image'] if i not in list(responses.image.unique())])
#print(responses.image.unique())  # Bloody.knife1  KKK.rally.1 


responses=responses.merge(imgnames,on="image",how="inner")
#print(picture_orders)

filenames=[filename for filename in os.listdir(imgfolder) if filename.endswith(".jpg")]
