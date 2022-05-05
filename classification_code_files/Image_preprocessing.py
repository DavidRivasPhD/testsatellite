import numpy as np
from osgeo import gdal, gdal_array
import pandas as pd
import operator
import gc
import os

roi_ds = gdal.Open('./Dataset/cdl2017.tiff', gdal.GA_ReadOnly)

#train_ds = gdal.Open('D:/neurafarms/downloaded_sat_images/rose_mlready/0306.tiff', gdal.GA_ReadOnly)

roi = roi_ds.GetRasterBand(1).ReadAsArray()

# How many pixels are in each class?
classes = np.unique(roi)


# Iterate over all class labels in the ROI image, printing out some information
# for c in classes:
#     print('Class {c} contains {n} pixels'.format(c=c,
#                                                  n=(roi == c).sum()))
dict = {}
for c in classes:
    dict[c] = (roi == c).sum()
sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
print("Top 6 classes and pixel counts \n",sorted_x[-6:])

#Select top  5 classes exclude 255 class label

top_classes = [69,75,36,121,225]
# Find how many non-zero entries we have -- i.e. how many training data samples?
n_samples = (roi > 0).sum()
print('We have {n} samples'.format(n=n_samples))

# What are our classification labels?
labels = np.unique(roi[roi > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                classes=labels))
# We will need a "X" matrix containing our features, and a "y" array containing our labels
#     These will have n_samples rows
#     In other languages we would need to allocate these and them loop to fill them, but NumPy can be faster

#X = img_b1[roi > 0, :]  
y = roi[roi > 0]
images = ['./Dataset/20170306.tiff',
          './Dataset/20170410.tiff',
          './Dataset/20170601.tiff',
          './Dataset/20170615.tiff',
          './Dataset/20170708.tiff',
          './Dataset/20170807.tiff',
          './Dataset/20170905.tiff',
          './Dataset/20170923.tiff',
          './Dataset/20171015.tiff',
          './Dataset/20171207.tiff']
		  
		  #69,75,36,121,225
print("Reading class 69")
final = pd.DataFrame()

for c in top_classes:
    
    temp = pd.DataFrame()
    
    print(c)
    
    for img in images:

        print(img)

        train_ds = gdal.Open(img, gdal.GA_ReadOnly)

        print(train_ds.RasterXSize,train_ds.RasterYSize)

        img_b1 = np.zeros((train_ds.RasterYSize, train_ds.RasterXSize, train_ds.RasterCount),
                       gdal_array.GDALTypeCodeToNumericTypeCode(train_ds.GetRasterBand(1).DataType))
        
        for b in range(img_b1.shape[2]):
            img_b1[:, :, b] = train_ds.GetRasterBand(b + 1).ReadAsArray()
        
        print(img_b1.shape)


        Xt = img_b1[roi==c, :] 
        
        Xt1 = pd.DataFrame(Xt)
        
        Xt2 = Xt1.sample(n=100000)
        
        Xt2.reset_index(drop=True,inplace=True)
        
        temp = pd.concat([Xt2,temp],axis=1)
        
        temp["class"] = c
        #temp.reset_index(drop=True,inplace=True)
      
    final = pd.concat([temp,final],axis=0)
    final.reset_index(drop=True,inplace=True)
    
    gc.collect()
	
	final


final.columns = ['col_'+str(i) for i in range(51)]
final.head()

final.to_csv("./Dataset/final.csv",index=False)
	
	