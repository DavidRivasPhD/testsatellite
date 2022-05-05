# Built-in modules
import os
import glob
import json
from typing import Tuple, List
from datetime import datetime, timedelta
import pickle
import shutil
import warnings
warnings.filterwarnings('ignore')

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm

# Data reding for training validation purposes:
from utils.data_transform import PlanetTransform, Sentinel1Transform, Sentinel2Transform
from utils.planet_reader import PlanetReader
from utils.sentinel_1_reader import S1Reader
from utils.sentinel_2_reader import S2Reader
from utils.data_loader import DataLoader
from utils.baseline_models import SpatiotemporalModel
from utils import train_valid_eval_utils as tveu
from utils import unzipper
from torch.optim import Adam
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn import NLLLoss
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix


brandenburg_tr_labels_dir='data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson'
brandenburg_te_labels_dir='data/dlr_fusion_competition_germany_test_labels/dlr_fusion_competition_germany_test_labels_33N_17E_243N/labels.geojson'
#CHECK TARGET DATA FORMAT IN TRAINING GROUND-TRUTH POLYGONS:

brandenburg_tr_labels=gpd.read_file(brandenburg_tr_labels_dir)
print('INFO: Number of fields: {}\n'.format(len(brandenburg_tr_labels)))
brandenburg_tr_labels.info()
brandenburg_tr_labels.tail()
#CHECK TARGET DATA FORMAT IN EVALUATION GROUND-TRUTH POLYGONS:

brandenburg_te_labels=gpd.read_file(brandenburg_te_labels_dir)
print('INFO: Number of fields: {}\n'.format(len(brandenburg_te_labels)))
brandenburg_te_labels.info()
brandenburg_te_labels.tail()
#CHECK LABEL IDs AND LABEL NAMES: 

label_ids=brandenburg_tr_labels['crop_id'].unique()
label_names=brandenburg_tr_labels['crop_name'].unique()

print('INFO: Label IDs: {}'.format(label_ids))
print('INFO: Label Names: {}'.format(label_names))
#CHECK FIELD DISTRIBUTION BY LABEL: 

value_counts=brandenburg_tr_labels['crop_name'].value_counts()

colors_list = ['#78C850','#A8B820','#F8D030','#E0C068', '#F08030', '#C03028', '#F85888','#6890F0','#98D8D8'] 
ax=value_counts.plot.bar(color=colors_list)
ax.set_ylabel("Number of Fields")
ax.set_xlabel("Crop Types")

print('INFO: Number of Fields by Crop Type: \n{}'.format(value_counts))
#CHECK TOTAL HECTARE DISTRIBUTION BY LABEL: 

hectare_distribution = pd.DataFrame(columns=["crop_name", "total_hectare"])
for name, group in brandenburg_tr_labels.groupby('crop_name'):
    total_hectare=group['SHAPE_AREA'].sum()/10000 # convert to m2 to hectare
    hectare_distribution=hectare_distribution.append({'crop_id':group.iloc[0]['crop_id'], 'crop_name':name, 'total_hectare':total_hectare}, ignore_index=True)

hectare_distribution.set_index('crop_id', inplace=True)
colors_list = ['#78C850','#A8B820','#F8D030','#E0C068', '#F08030', '#C03028', '#F85888','#6890F0','#98D8D8'] 
ax=hectare_distribution.plot.barh(color=colors_list,x='crop_name', y='total_hectare',legend=False)
ax.set_xlabel("Total Hectare per Crop Type")
ax.set_ylabel("Crop Types")

print('INFO: Total Hectare per Crop Type: \n{}'.format(hectare_distribution.sort_index()))
#CHECK HECTARE DISTRIBUTION HISTOGRAM BY LABEL: 

#Convert m2 to hectare:
histogram_data = brandenburg_tr_labels.copy(deep=True)
histogram_data['SHAPE_AREA']=brandenburg_tr_labels['SHAPE_AREA']/10000

ax=histogram_data.hist( by='crop_name',column = 'SHAPE_AREA', bins=25,figsize=(16,16))
for i in range(ax.shape[0]): 
    for j in range(ax.shape[1]): 
        ax[i][j].set_ylabel("Number of fields with the given hectare size")
        ax[i][j].set_xlabel("Hectare")
        #DISPLAY TARGET FIELDS of 2018 FOR TRAINING ON THE MAP BY LABEL: 

fig, ax = plt.subplots(figsize=(18, 18))
counter=0
legend_elements = []
for name, group in brandenburg_tr_labels.groupby('crop_name'):
    group.plot(ax=ax,color=colors_list[counter], aspect=1)
    legend_elements.append(Patch(facecolor=colors_list[counter], edgecolor=colors_list[counter],label=name))
    counter+=1

ax.legend(handles=legend_elements,loc='lower right')
ax.title.set_text('BRANDENBURG 2018: GROUND TRUTH POLYGONS WITH CROP LABELS for TRAINING')
#DISPLAY TARGET FIELDS of 2019 WITHOUT LABELS : 

brandenburg_te=gpd.read_file(brandenburg_te_labels_dir)
fig, ax = plt.subplots(figsize=(18, 18))
counter=0
legend_elements = []
for name, group in brandenburg_te.groupby('crop_name'):
    group.plot(ax=ax,color=colors_list[counter], aspect=1)
    legend_elements.append(Patch(facecolor=colors_list[counter], edgecolor=colors_list[counter],label=name))
    counter+=1

ax.legend(handles=legend_elements,loc='lower right')
ax.title.set_text('BRANDENBURG 2019: GROUND TRUTH POLYGONS WITHOUT CROP LABELS for the EVALUATION')
#DIRECTORY OF PLANET FUSION TRAINING DATA AND GROUND TRUTHS:

brandenburg_planet_train_dir='data/dlr_fusion_competition_germany_train_source_planet_5day/'
brandenburg_tr_labels_dir='data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson'
#INITIALIZE THE DATA READER TO OBSERVE THE FIELDS FROM PLANET DATA: 

# Choose some days of the year to plot
selected_days_of_year = [10,20,30,40, 50] #from 365 days of the year

#Initialize data reader for planet images
planet_reader = PlanetReader(input_dir=brandenburg_planet_train_dir,
                                  label_dir=brandenburg_tr_labels_dir,
                                  selected_time_points=selected_days_of_year)
								  #DEFINE TRUE COLOR IMAGING AND NDVI INDEXING FUNCTIONS FOR VISUALISATION OF PLANET DATA: 

#Define NDVI index for Planet Fusion images
def ndvi(X):
    red = X[2]
    nir = X[3]
    return (nir-red) / (nir + red)

#Define True Color for Planet Fusion images
def true_color(X):
    blue = X[0]/(X[0].max()/255.0)
    green = X[1]/(X[1].max()/255.0)
    red = X[2]/(X[2].max()/255.0)
    tc = np.dstack((red,green,blue)) 
    
    return tc.astype('uint8')
	#VISUALISE SOME OF THE FIELDS FROM PLANET DATA: 

#Initialize plot cells
num_row = 2 * len(selected_days_of_year)
num_col = len(label_ids)
fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))

#Display one sample field for each crop type
pbar = tqdm(total=len(label_ids))
iterable=iter(planet_reader)
for crop_id, crop_name in zip(label_ids,label_names):
    while True:
        # read a field sample
        X,y,mask,_ = next(iterable) 
        
        width=X.shape[-1]
        height=X.shape[-2]
        
        #Get one sample for each crop type, and
        #consider large areas (at least 200x200) to display
        if y == crop_id and width>200 and height>200:
            for i, day in enumerate(selected_days_of_year):
                
                # Display RGB image of the field in a given week for a given crop type
                ax = axes[(2*i)%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('RGB in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(true_color(X[i]))
                
                # Display NDVI index of the field in a given day for a given crop type
                ax = axes[(2*i+1)%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('NDVI in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(ndvi(X[i]*mask),cmap=plt.cm.summer)
                
            #if one sample selected for a crop type, break the WHILE loop
            pbar.set_description("Plotting {} Fields".format(crop_name))
            pbar.update(1)
            break
        
        
plt.tight_layout()
plt.show()
pbar.set_description("Plotting Complete!")
pbar.close()
#DIRECTORY OF SENTINEL-1 TRAINING DATA :
brandenburg_s1_asc_train_dir = "data/dlr_fusion_competition_germany_train_source_sentinel_1/dlr_fusion_competition_germany_train_source_sentinel_1_asc_33N_18E_242N_2018/" #ASCENDING ORBIT
brandenburg_s1_dsc_train_dir = "data/dlr_fusion_competition_germany_train_source_sentinel_1/dlr_fusion_competition_germany_train_source_sentinel_1_ds
#INITIALIZE THE DATA READER TO OBSERVE THE FIELDS FROM S1 DATA: 

# Choose some days of the year to plot
selected_data_indices = [1,8,15,22,29] #beware that S1 data is not daily, 

#Initialize data reader for planet images
s1_reader = S1Reader(input_dir=brandenburg_s1_asc_train_dir,
                                  label_dir=brandenburg_tr_labels_dir,
                                  selected_time_points=selected_data_indices)
								  # DEFINE RADAR VEGETATION INDEXING FOR VISUALISATION OF  S1 DATA: 
# for the algorithm please refer to Sentinel Hub: 
# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-1/radar_vegetation_index_code_dual_polarimetric/

def rvi(X):
    VV = X[0]
    VH = X[1]
    dop = (VV/(VV+VH))
    m = 1 - dop
    radar_vegetation_index = (np.sqrt(dop))*((4*(VH))/(VV+VH))
    
    return radar_vegetation_index
	#VISUALISE SOME OF THE FIELDS FROM S1 DATA: 

#Initialize plot cells
num_row = len(selected_data_indices)
num_col = len(label_ids)
fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))

#Display one sample field for each crop type
pbar = tqdm(total=len(label_ids))
iterable=iter(s1_reader)
for crop_id, crop_name in zip(label_ids,label_names):
    while True:
        # read a field sample
        X,y,mask,_ = next(iterable) 
        
        width=X.shape[-1]
        height=X.shape[-2]
        
        #Get one sample for each crop type, and
        #consider large areas (at least 60x60) to display
        if y == crop_id and width>60 and height>60:
            for i, day in enumerate(selected_data_indices):
                 
                # Display RVI index of the field in a given day for a given crop type
                ax = axes[i%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('RVI in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(rvi(X[i]*mask),cmap=plt.cm.summer)
                
            #if one sample selected for a crop type, break the WHILE loop
            pbar.set_description("Plotting {} Fields".format(crop_name))
            pbar.update(1)
            break
        
        
plt.tight_layout()
plt.show()
pbar.set_description("Plotting Complete!")
pbar.close()
#DIRECTORY OF SENTINEL-2 TRAINING DATA :

brandenburg_s2_train_dir = "data/dlr_fusion_competition_germany_train_source_sentinel_2/dlr_fusion_competition_germany_train_source_sentinel_2_33N_18E_242N_2018/"
#INITIALIZE THE DATA READER TO OBSERVE THE FIELDS FROM PLANET DATA: 

# Choose some days of the year to plot
selected_data_indices = [10,20,30,40,50] #beware that S2 data is not daily, 

#Initialize data reader for planet images
s2_reader = S2Reader(input_dir=brandenburg_s2_train_dir,
                                  label_dir=brandenburg_tr_labels_dir,
                                  selected_time_points=selected_data_indices)
								  south_africa_tr_labels_dir_1='data/ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson'
south_africa_tr_labels_dir_2='data/ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_259N/labels.geojson'
south_africa_te_labels_dir = 'data/ref_fusion_competition_south_africa_test_labels/ref_fusion_competition_south_africa_test_labels_34S_20E_259N/labels
#CHECK TARGET DATA FORMAT IN TRAINING GROUND-TRUTH POLYGONS:

south_africa_tr_labels_1=gpd.read_file(south_africa_tr_labels_dir_1)
print('INFO: Number of fields: {}\n'.format(len(south_africa_tr_labels_1)))
south_africa_tr_labels_1.info()
south_africa_tr_labels_1.tail()
#CHECK TARGET DATA FORMAT IN TRAINING GROUND-TRUTH POLYGONS:

south_africa_tr_labels_2=gpd.read_file(south_africa_tr_labels_dir_2)
print('INFO: Number of fields: {}\n'.format(len(south_africa_tr_labels_2)))
south_africa_tr_labels_2.info()
south_africa_tr_labels_2.tail()
#CHECK TARGET DATA FORMAT IN EVALUATION GROUND-TRUTH POLYGONS:

south_africa_te_labels=gpd.read_file(south_africa_te_labels_dir)
print('INFO: Number of fields: {}\n'.format(len(south_africa_te_labels)))
south_africa_te_labels.info()
south_africa_te_labels.tail()

#CHECK LABEL IDs AND LABEL NAMES: 

label_ids=south_africa_tr_labels_2['crop_id'].unique()
label_names=south_africa_tr_labels_2['crop_name'].unique()

print('INFO: Label IDs: {}'.format(label_ids))
print('INFO: Label Names: {}'.format(label_names))

These plant types are not evenly planted in the agricultural fields, so you can observe the distribution of fields with each particular crop as below:

#CHECK FIELD DISTRIBUTION BY LABEL: 

value_counts =south_africa_tr_labels_1['crop_name'].value_counts()
value_counts += south_africa_tr_labels_2['crop_name'].value_counts()

colors_list = ['#78C850','#A8B820','#F8D030','#E0C068', '#F08030', '#C03028', '#F85888','#6890F0','#98D8D8'] 
ax=value_counts.plot.bar(color=colors_list)
ax.set_ylabel("Number of Fields")
ax.set_xlabel("Crop Types")

print('INFO: Number of Fields by Crop Type: \n{}'.format(value_counts))
#CHECK TOTAL HECTARE DISTRIBUTION BY LABEL: 

hectare_distribution = pd.DataFrame(columns=["crop_name", "total_hectare"])
for group_1, group_2 in zip(south_africa_tr_labels_1.groupby('crop_name'),
                                        south_africa_tr_labels_2.groupby('crop_name')):
    crop_id=group_1[1].iloc[0]['crop_id']
    crop_name=group_1[0]
    total_hectare_1=group_1[1]['SHAPE_AREA'].sum()/10000 # convert to m2 to hectare
    total_hectare_2=group_2[1]['SHAPE_AREA'].sum()/10000 # convert to m2 to hectare
    total_hectare= total_hectare_1 + total_hectare_2
    
    hectare_distribution=hectare_distribution.append({'crop_id':crop_id, 'crop_name':crop_name, 'total_hectare':total_hectare}, ignore_index=True)

hectare_distribution.set_index('crop_id', inplace=True)
colors_list = ['#78C850','#A8B820','#F8D030','#E0C068', '#F08030', '#C03028', '#F85888','#6890F0','#98D8D8'] 
ax=hectare_distribution.plot.barh(color=colors_list,x='crop_name', y='total_hectare',legend=False)
ax.set_xlabel("Total Hectare per Crop Type")
ax.set_ylabel("Crop Types")

print('INFO: Total Hectare per Crop Type: \n{}'.format(hectare_distribution.sort_index()))
#DISPLAY TARGET FIELDS AT TILES '19E-258N' and '19E-259N' FOR TRAINING ON THE MAP BY LABEL: 

fig, axes = plt.subplots(1,2, figsize=(18, 18))
counter=0
legend_elements = []
for group_1, group_2 in zip(south_africa_tr_labels_1.groupby('crop_name'),
                            south_africa_tr_labels_2.groupby('crop_name')):
    
    group_1[1].plot(ax=axes[0],color=colors_list[counter], aspect=1)
    group_2[1].plot(ax=axes[1],color=colors_list[counter], aspect=1)
    
    legend_elements.append(Patch(facecolor=colors_list[counter], edgecolor=colors_list[counter],label=group_1[0]))
    counter+=1


axes[0].legend(handles=legend_elements,loc='lower right')
axes[0].title.set_text('SOUTH AFRICA 2017: GROUND TRUTH POLYGONS AT 19E-258N WITH CROP LABELS')

axes[1].legend(handles=legend_elements,loc='lower right')
axes[1].title.set_text('SOUTH AFRICA 2017: GROUND TRUTH POLYGONS AT 19E-259N WITH CROP LABELS')

#DISPLAY TARGET FIELDS AT TILES '20E-259N' WITHOUT LABELS : 

south_africa_te_labels=gpd.read_file(south_africa_te_labels_dir)
fig, ax = plt.subplots(figsize=(18, 18))
counter=0
legend_elements = []
for name, group in south_africa_te_labels.groupby('crop_name'):
    group.plot(ax=ax,color=colors_list[counter], aspect=1)
    legend_elements.append(Patch(facecolor=colors_list[counter], edgecolor=colors_list[counter],label=name))
    counter+=1

ax.legend(handles=legend_elements,loc='lower right')
ax.title.set_text('SOUTH AFRICA 2017: GROUND TRUTH POLYGONS AT 20E-259N WITHOUT CROP LABELS')

#DIRECTORY OF PLANET FUSION TRAINING DATA AND LABELS:

south_africa_planet_train_dir_1='data/ref_fusion_competition_south_africa_train_source_planet_5day/'
south_africa_tr_labels_dir_1='data/ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson'



south_africa_planet_train_dir_2='data/ref_fusion_competition_south_africa_train_source_planet_5day'
south_africa_tr_labels_dir_2='data/ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_259N/labels.geojson'
#INITIALIZE THE DATA READER TO OBSERVE THE FIELDS FROM PLANET DATA: 

# Choose some days of the year to plot
selected_days_of_year = [1, 2, 3, 4, 5] #from 244 days of the year

#Initialize data reader for planet images
planet_reader = PlanetReader(input_dir=south_africa_planet_train_dir_1,
                                  label_dir=south_africa_tr_labels_dir_1,
                                  selected_time_points=selected_days_of_year)
								  
								  #VISUALISE SOME OF THE FIELDS FROM PLANET DATA: 

#Initialize plot cells
num_row = 2 * len(selected_days_of_year)
num_col = len(label_ids)
fig, axes = plt.subplots(num_row, num_col, figsize=(2.5*num_col,2*num_row))

#Display one sample field for each crop type
pbar = tqdm(total=len(label_ids))
iterable=iter(planet_reader)
for crop_id, crop_name in zip(label_ids,label_names):
    while True:
        # read a field sample
        X,y,mask,fid = next(iterable) 
        
        width=X.shape[-1]
        height=X.shape[-2]
        
        #Get one sample for each crop type, and
        #consider large areas (at least 200x200) to display
        if y == crop_id and width>200 and height>200:
            for i, day in enumerate(selected_days_of_year):
                
                # Display RGB image of the field in a given week for a given crop type
                ax = axes[(2*i)%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('RGB in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(true_color(X[i]))
                
                # Display NDVI index of the field in a given day for a given crop type
                ax = axes[(2*i+1)%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('NDVI in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(ndvi(X[i]*mask),cmap=plt.cm.summer)
                
            #if one sample selected for a crop type, break the WHILE loop
            pbar.set_description("Plotting {} Fields".format(crop_name))
            pbar.update(1)
            break
        
        
plt.tight_layout()
plt.show()
pbar.set_description("Plotting Complete!")
pbar.close()

#DIRECTORY OF SENTINEL-1 TRAINING DATA:

south_africa_s1_asc_train_dir_1 = "data/ref_fusion_competition_south_africa_train_source_sentinel_1/ref_fusion_competition_south_africa_train_source_sentinel_1_34S_19E_258N_34S_19E_258N_2017/"
south_africa_s1_asc_train_dir_2 = "data/ref_fusion_competition_south_africa_train_source_sentinel_1/ref_fusion_competition_south_africa_train_source_sentinel_1_34S_19E_259N_34S_19E_259N_2017/"
#INITIALIZE THE DATA READER TO OBSERVE THE FIELDS FROM S1 DATA: 

# Choose some days of the year to plot
selected_data_indices = [10,15,20,25,30] #beware that S1 data is not daily, 

#Initialize data reader for planet images
s1_reader = S1Reader(input_dir=south_africa_s1_asc_train_dir_1,
                                  label_dir=south_africa_tr_labels_dir_1,
                                  selected_time_points=selected_data_indices)
								  
								  #VISUALISE SOME OF THE FIELDS FROM S1 DATA: 

#Initialize plot cells
num_row = len(selected_data_indices)
num_col = len(label_ids)
fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))

#Display one sample field for each crop type
pbar = tqdm(total=len(label_ids))
iterable=iter(s1_reader)
for crop_id, crop_name in zip(label_ids,label_names):
    while True:
        # read a field sample
        X,y,mask,fid = next(iterable) 
        width=X.shape[-1]
        height=X.shape[-2]
        
        #Get one sample for each crop type, and
        #consider large areas (at least 60x60) to display
        if y == crop_id and width>60 and height>60:
            for i, day in enumerate(selected_data_indices):
                 
                # Display RVI index of the field in a given day for a given crop type
                ax = axes[i%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('RVI in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(rvi(X[i]*mask),cmap=plt.cm.summer)
                
            #if one sample selected for a crop type, break the WHILE loop
            pbar.set_description("Plotting {} Fields".format(crop_name))
            pbar.update(1)
            break
        
        
plt.tight_layout()
plt.show()
pbar.set_description("Plotting Complete!")
pbar.close()

#DIRECTORY OF SENTINEL-2 TRAINING DATA:

south_africa_s2_train_dir_1 = "data/ref_fusion_competition_south_africa_train_source_sentinel_2/ref_fusion_competition_south_africa_train_source_sentinel_2_34S_19E_258N_34S_19E_258N_2017/"
south_africa_s2_train_dir_2 = "data/ref_fusion_competition_south_africa_train_source_sentinel_2/ref_fusion_competition_south_africa_train_source_sentinel_2_34S_19E_259N_34S_19E_259N_2017/"
#INITIALIZE THE DATA READER TO OBSERVE THE FIELDS FROM S2 DATA: 

# Choose some days of the year to plot
selected_data_indices = [12,18,24,30,36] #beware that S2 data is not daily, 

#Initialize data reader for planet images
s2_reader = S2Reader(input_dir=south_africa_s2_train_dir_1,
                                  label_dir=south_africa_tr_labels_dir_1,
                                  selected_time_points=selected_data_indices)
								  
								  #DEFINE TRUE COLOR IMAGING AND NDVI INDEXING FUNCTIONS FOR VISUALISATION OF S2 DATA: 

#Define NDVI index for S2 images
def ndvi(X):
    red = X[3]
    nir = X[7]
    return (nir-red) / (nir + red)

#Define True Color for S2 images
def true_color(X):
    blue = X[1]/(X[1].max()/255.0)
    green = X[2]/(X[2].max()/255.0)
    red = X[3]/(X[3].max()/255.0)
    tc = np.dstack((red,green,blue)) 
    
    return tc.astype('uint8')
#VISUALISE SOME OF THE FIELDS FROM S2 DATA: 

#Initialize plot cells
num_row = 2 * len(selected_days_of_year)
num_col = len(label_ids)
fig, axes = plt.subplots(num_row, num_col, figsize=(2.5*num_col,2*num_row))

#Display one sample field for each crop type
pbar = tqdm(total=len(label_ids))
iterable=iter(s2_reader)
for crop_id, crop_name in zip(label_ids,label_names):
    while True:
        # read a field sample
        X,y,mask,fid = next(iterable) 
        
        width=X.shape[-1]
        height=X.shape[-2]
        
        #Get one sample for each crop type, and
        #consider large areas (at least 60x60) to display
        if y == crop_id and width>60 and height>60:
            for i, day in enumerate(selected_data_indices):
                
                # Display RGB image of the field in a given week for a given crop type
                ax = axes[(2*i)%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('RGB in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(true_color(X[i]))
                
                # Display NDVI index of the field in a given day for a given crop type
                ax = axes[(2*i+1)%num_row, crop_id%num_col]
                ax.title.set_text('{}'.format(crop_name))
                ax.set_ylabel('NDVI in Day {}'.format(day))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(ndvi(X[i]*mask),cmap=plt.cm.summer)
                
            #if one sample selected for a crop type, break the WHILE loop
            pbar.set_description("Plotting {} Fields".format(crop_name))
            pbar.update(1)
            break
        
        
plt.tight_layout()
plt.show()
pbar.set_description("Plotting Complete!")
pbar.close()

#LET'S RECALL NECESSARY DIRECTORIES FOR TRAINING ON BRANDENBURG AREA: 

brandenburg_planet_train_dir='data/dlr_fusion_competition_germany_train_source_planet_5day/'
brandenburg_tr_labels_dir='data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson'

brandenburg_tr_labels=gpd.read_file(brandenburg_tr_labels_dir)

label_ids=brandenburg_tr_labels['crop_id'].unique()
label_names=brandenburg_tr_labels['crop_name'].unique()
#SORT LABEL IDS and NAMES
zipped_lists = zip(label_ids, label_names)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
label_ids, label_names = [ list(tuple) for tuple in  tuples]
#INITIALIZE DATA LOADERS FOR TRAINING AND EVALUATION: 

#Get data transformer for planet images
planet_transformer=PlanetTransform()

#Initialize data reader for planet images
planet_reader = PlanetReader(input_dir=brandenburg_planet_train_dir,
                             label_dir=brandenburg_tr_labels_dir,
                             label_ids=label_ids,
                             transform=planet_transformer.transform,
                             min_area_to_ignore=1000)

#Initialize data loaders
data_loader=DataLoader(train_val_reader=planet_reader, validation_split=0.25)
train_loader=data_loader.get_train_loader(batch_size=8, num_workers=1)
valid_loader=data_loader.get_validation_loader(batch_size=8, num_workers=1)


#INITIALIZE TRAINING MODEL: 

INPUT_DIM=4  # number of channels in Planet Fusion data
SEQUENCE_LENGTH=74  #Sequence size of Planet Fusion Data
DEVICE='cpu' 
START_EPOCH=0
TOTAL_EPOCH=1

model = SpatiotemporalModel(input_dim=INPUT_DIM, num_classes=len(label_ids), sequencelength=SEQUENCE_LENGTH, device=DEVICE)
    
# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1e-2
for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

#INITIALIZE MODEL OPTIMIZER AND LOSS CRITERION: 

#Initialize model optimizer and loss criterion:
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
loss_criterion = CrossEntropyLoss(reduction="mean")
#SET LOG DIRECTORY FOR THE TRAINING AND VALIDATION OUTPUTS:

log = list()
log_root='temp_planet/'
logdir = os.path.join(log_root, model.modelname)
os.makedirs(logdir, exist_ok=True)
print("INFO: Logging results will be saved to {}".format(logdir))
summarywriter = SummaryWriter(log_dir=logdir)
snapshot_path = os.path.join(logdir, "model.pth.tar")

#IF THERE IS ALREADY A TRAINED MODEL, RESUME IT:

snapshot_path = os.path.join(logdir, "model.pth.tar")
if os.path.exists(snapshot_path):
    checkpoint = torch.load(snapshot_path)
    START_EPOCH = checkpoint["epoch"]
    log = checkpoint["log"]
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    model.load_state_dict(checkpoint["model_state"])
    print(f"INFO: Resuming from {snapshot_path}, epoch {START_EPOCH}")
	
	#DEFINE TRAINING AND VALIDATION EPOCH FUNCTIONS:


for epoch in range(START_EPOCH, TOTAL_EPOCH):
    train_loss = tveu.train_epoch(model, optimizer, loss_criterion, train_loader, device=DEVICE)
    valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(model, loss_criterion, valid_loader, device=DEVICE)
    
    
    scores = tveu.metrics(y_true.cpu(), y_pred.cpu())
    
    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
    
    valid_loss = valid_loss.cpu().detach().numpy()[0]
    train_loss = train_loss.cpu().detach().numpy()[0]

    scores["epoch"] = epoch
    scores["train_loss"] = train_loss
    scores["valid_loss"] = valid_loss
    log.append(scores)

    summarywriter.add_scalars("losses", dict(train=train_loss, valid=valid_loss), global_step=epoch)
    summarywriter.add_scalars("metrics",
                              {key: scores[key] for key in
                               ['accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 
                                'recall_micro','recall_macro', 'recall_weighted', 
                                'precision_micro', 'precision_macro','precision_weighted']},
                                global_step=epoch)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred.cpu().detach().numpy(), labels=np.arange(len(label_ids)))
    summarywriter.add_figure("confusion_matrix",tveu.confusion_matrix_figure(cm, labels=label_ids),global_step=epoch)

    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(logdir, "train_log.csv"))

    torch.save(dict( model_state=model.state_dict(),optimizer_state=optimizer.state_dict(), epoch=epoch, log=log),snapshot_path)
    if len(log) > 2:
        if valid_loss < np.array([l["valid_loss"] for l in log[:-1]]).min():
            best_model = snapshot_path.replace("model.pth.tar","model_best.pth.tar")
            print(f"INFO: New best model with valid_loss {valid_loss:.2f} at {best_model}")
            shutil.copy(snapshot_path, best_model)

    print(f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} " + scores_msg)
	
	#LET'S RECALL NECESSARY DIRECTORIES FOR TRAINING ON A TILE OF BRANDENBURG: 
brandenburg_s1_train_dir = "data/dlr_fusion_competition_germany_train_source_sentinel_1/dlr_fusion_competition_germany_train_source_sentinel_1_asc_33N_18E_242N_2018/" #ASCENDING ORBIT
brandenburg_tr_labels_dir= "data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson"

brandenburg_tr_labels=gpd.read_file(brandenburg_tr_labels_dir)
label_ids=brandenburg_tr_labels['crop_id'].unique()
label_names=brandenburg_tr_labels['crop_name'].unique()
#SORT LABEL IDS and NAMES
zipped_lists = zip(label_ids, label_names)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
label_ids, label_names = [ list(tuple) for tuple in  tuples]
#INITIALIZE DATA LOADERS FOR TRAINING AND EVALUATION: 

#Get data transformer for S1 images
sentinel_1_transformer=Sentinel1Transform(normalize=True, image_size=32)

#Initialize data reader for S1 images
s1_reader = S1Reader(input_dir=brandenburg_s1_train_dir,
                             label_dir=brandenburg_tr_labels_dir,
                             label_ids=label_ids,
                             transform=sentinel_1_transformer.transform,
                             min_area_to_ignore=1000)

#Initialize data loaders
data_loader=DataLoader(train_val_reader=s1_reader, validation_split=0.25)
train_loader=data_loader.get_train_loader(batch_size=8, num_workers=1)
valid_loader=data_loader.get_validation_loader(batch_size=8, num_workers=1)


#INITIALIZE TRAINING MODEL: 

INPUT_DIM=2  # number of channels in S1 data
SEQUENCE_LENGTH=77  #Sequence size of Temporal Data
DEVICE='cpu' 
START_EPOCH=0
TOTAL_EPOCH=1

brandenburg_model = SpatiotemporalModel(input_dim=INPUT_DIM, num_classes=len(label_ids), sequencelength=SEQUENCE_LENGTH, device=DEVICE)
    
# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1
for p in brandenburg_model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

#INITIALIZE MODEL OPTIMIZER AND LOSS CRITERION: 

#Initialize model optimizer and loss criterion:
optimizer = SGD(brandenburg_model.parameters(), lr=1e-3, momentum=0.9,nesterov=False)
loss_criterion = CrossEntropyLoss()
#SET LOG DIRECTORY FOR THE TRAINING AND VALIDATION OUTPUTS:

log = list()
log_root='temp_s1/'
logdir = os.path.join(log_root, brandenburg_model.modelname)
os.makedirs(logdir, exist_ok=True)
print("INFO: Logging results will be saved to {}".format(logdir))
summarywriter = SummaryWriter(log_dir=logdir)
snapshot_path = os.path.join(logdir, "model.pth.tar")

#IF THERE IS ALREADY A TRAINED MODEL, RESUME IT:

snapshot_path = os.path.join(logdir, "model.pth.tar")
if os.path.exists(snapshot_path):
    checkpoint = torch.load(snapshot_path)
    START_EPOCH = checkpoint["epoch"]
    log = checkpoint["log"]
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    brandenburg_model.load_state_dict(checkpoint["model_state"])
    print(f"INFO: Resuming from {snapshot_path}, epoch {START_EPOCH}")

#DEFINE TRAINING AND VALIDATION EPOCH FUNCTIONS:


for epoch in range(START_EPOCH, TOTAL_EPOCH):
    train_loss = tveu.train_epoch(brandenburg_model, optimizer, loss_criterion, train_loader, device=DEVICE)
    valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(brandenburg_model, loss_criterion, valid_loader, device=DEVICE)
    
    
    scores = tveu.metrics(y_true.cpu(), y_pred.cpu())
    
    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
    
    valid_loss = valid_loss.cpu().detach().numpy()[0]
    train_loss = train_loss.cpu().detach().numpy()[0]

    scores["epoch"] = epoch
    scores["train_loss"] = train_loss
    scores["valid_loss"] = valid_loss
    log.append(scores)

    summarywriter.add_scalars("losses", dict(train=train_loss, valid=valid_loss), global_step=epoch)
    summarywriter.add_scalars("metrics",
                              {key: scores[key] for key in
                               ['accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 
                                'recall_micro','recall_macro', 'recall_weighted', 
                                'precision_micro', 'precision_macro','precision_weighted']},
                                global_step=epoch)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred.cpu().detach().numpy(), labels=np.arange(len(label_ids)))
    summarywriter.add_figure("confusion_matrix",tveu.confusion_matrix_figure(cm, labels=label_ids),global_step=epoch)

    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(logdir, "train_log.csv"))

    torch.save(dict( model_state=brandenburg_model.state_dict(),optimizer_state=optimizer.state_dict(), epoch=epoch, log=log),snapshot_path)
    if len(log) > 2:
        if valid_loss < np.array([l["valid_loss"] for l in log[:-1]]).min():
            best_model = snapshot_path.replace("model.pth.tar","model_best.pth.tar")
            print(f"INFO: New best model with valid_loss {valid_loss:.2f} at {best_model}")
            shutil.copy(snapshot_path, best_model)

    print(f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} " + scores_msg)
	
	#LET'S RECALL NECESSARY DIRECTORIES FOR TRAINING ON A TILE OF SOUTH AFRICA: 


south_africa_s2_train_dir_1 = "data/ref_fusion_competition_south_africa_train_source_sentinel_2/ref_fusion_competition_south_africa_train_source_sentinel_2_34S_19E_258N_34S_19E_258N_2017/"
south_africa_tr_labels_dir_1= "data/ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_258N/labels.geojson"

south_africa_s2_train_dir_2 = "data/ref_fusion_competition_south_africa_train_source_sentinel_2/ref_fusion_competition_south_africa_train_source_sentinel_2_34S_19E_259N_34S_19E_259N_2017/"
south_africa_tr_labels_dir_2= "data/ref_fusion_competition_south_africa_train_labels/ref_fusion_competition_south_africa_train_labels_34S_19E_259N/labels.geojson"


south_africa_tr_labels_1=gpd.read_file(south_africa_tr_labels_dir_1)
label_ids=south_africa_tr_labels_1['crop_id'].unique()
label_names=south_africa_tr_labels_1['crop_name'].unique()
#SORT LABEL IDS and NAMES
zipped_lists = zip(label_ids, label_names)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
label_ids, label_names = [ list(tuple) for tuple in  tuples]
#INITIALIZE DATA LOADERS FOR TRAINING AND EVALUATION: 

#Get data transformer for S2 images
sentinel_2_transformer=Sentinel2Transform()

#Initialize data reader for S2 images
s2_reader = S2Reader(input_dir=south_africa_s2_train_dir_1,
                             label_dir=south_africa_tr_labels_dir_1,
                             label_ids=label_ids,
                             transform=sentinel_2_transformer.transform,
                             min_area_to_ignore=1000)

#Initialize data loaders
data_loader=DataLoader(train_val_reader=s2_reader, validation_split=0.25)
train_loader=data_loader.get_train_loader(batch_size=8, num_workers=1)
valid_loader=data_loader.get_validation_loader(batch_size=8, num_workers=1)



#INITIALIZE TRAINING MODEL: 

INPUT_DIM=12  # number of channels in S2 data
SEQUENCE_LENGTH=50  #Sequence size of Temporal Data
DEVICE='cpu' 
START_EPOCH=0
TOTAL_EPOCH=1

africa_model = SpatiotemporalModel(input_dim=INPUT_DIM, num_classes=len(label_ids), sequencelength=SEQUENCE_LENGTH, device=DEVICE)
    
# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1e-2
for p in africa_model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

#INITIALIZE MODEL OPTIMIZER AND LOSS CRITERION: 

#Initialize model optimizer and loss criterion:
optimizer = Adam(africa_model.parameters(), lr=1e-3, weight_decay=1e-6)
loss_criterion = CrossEntropyLoss(reduction="mean")
#SET LOG DIRECTORY FOR THE TRAINING AND VALIDATION OUTPUTS:

log = list()
log_root='temp_s2/'
logdir = os.path.join(log_root, africa_model.modelname)
os.makedirs(logdir, exist_ok=True)
print("INFO: Logging results will be saved to {}".format(logdir))
summarywriter = SummaryWriter(log_dir=logdir)
snapshot_path = os.path.join(logdir, "model.pth.tar")




#IF THERE IS ALREADY A TRAINED MODEL, RESUME IT:

snapshot_path = os.path.join(logdir, "model.pth.tar")
if os.path.exists(snapshot_path):
    checkpoint = torch.load(snapshot_path)
    START_EPOCH = checkpoint["epoch"]
    log = checkpoint["log"]
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    africa_model.load_state_dict(checkpoint["model_state"])
    print(f"INFO: Resuming from {snapshot_path}, epoch {START_EPOCH}")
-
#DEFINE TRAINING AND VALIDATION EPOCH FUNCTIONS:


for epoch in range(START_EPOCH, TOTAL_EPOCH):
    train_loss = tveu.train_epoch(africa_model, optimizer, loss_criterion, train_loader, device=DEVICE)
    valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(africa_model, loss_criterion, valid_loader, device=DEVICE)
    
    
    scores = tveu.metrics(y_true.cpu(), y_pred.cpu())
    
    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
    
    valid_loss = valid_loss.cpu().detach().numpy()[0]
    train_loss = train_loss.cpu().detach().numpy()[0]

    scores["epoch"] = epoch
    scores["train_loss"] = train_loss
    scores["valid_loss"] = valid_loss
    log.append(scores)

    summarywriter.add_scalars("losses", dict(train=train_loss, valid=valid_loss), global_step=epoch)
    summarywriter.add_scalars("metrics",
                              {key: scores[key] for key in
                               ['accuracy', 'kappa', 'f1_micro', 'f1_macro', 'f1_weighted', 
                                'recall_micro','recall_macro', 'recall_weighted', 
                                'precision_micro', 'precision_macro','precision_weighted']},
                                global_step=epoch)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred.cpu().detach().numpy(), labels=np.arange(len(label_ids)))
    summarywriter.add_figure("confusion_matrix",tveu.confusion_matrix_figure(cm, labels=label_ids),global_step=epoch)

    log_df = pd.DataFrame(log).set_index("epoch")
    log_df.to_csv(os.path.join(logdir, "train_log.csv"))

    torch.save(dict( model_state=africa_model.state_dict(),optimizer_state=optimizer.state_dict(), epoch=epoch, log=log),snapshot_path)
    if len(log) > 2:
        if valid_loss < np.array([l["valid_loss"] for l in log[:-1]]).min():
            best_model = snapshot_path.replace("model.pth.tar","model_best.pth.tar")
            print(f"INFO: New best model with valid_loss {valid_loss:.2f} at {best_model}")
            shutil.copy(snapshot_path, best_model)

    print(f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} " + scores_msg)