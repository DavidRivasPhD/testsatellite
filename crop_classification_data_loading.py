Imports
#general
import os
import shutil

#plotting
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

#progressbar
from tqdm.auto import tqdm


import geopandas as gpd

from radiant_mlhub import get_session
from radiant_mlhub import Dataset,Collection
Download labels

#get key
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("RadiantEarth_MLHub")

#authenticate
os.environ['MLHUB_API_KEY'] = secret_value_0
session = get_session()



# Download the Brandenburg-Germany Dataset
dataset = Dataset.fetch('dlr_fusion_competition_germany')


for file in dataset.collections:
    print(f'{file.id}: {file.title}....Size in GB is {file.archive_size/(1024**3)}')
    Downloading Labels

labels_dir = './labels'

if not os.path.exists(labels_dir):
    os.mkdir(labels_dir)
    
#dowload train labels
lab_tr = Collection.fetch('dlr_fusion_competition_germany_train_labels')
print(f'train labels size in MB {lab_tr.archive_size/ 1024**2}')
#download
lab_tr.download(labels_dir)


#dowload test labels
lab_ts = Collection.fetch('dlr_fusion_competition_germany_test_labels')
print(f'test labels size in MB {lab_tr.archive_size/ 1024**2}')

#download
lab_ts.download(labels_dir)
unpack labels
shutil.unpack_archive('./labels/dlr_fusion_competition_germany_test_labels.tar.gz')
shutil.unpack_archive('./labels/dlr_fusion_competition_germany_train_labels.tar.gz')

#remove labels archive
shutil.rmtree(labels_dir)
Load data

#labels
tr_labels_dir = './dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson'
ts_labels_dir = './dlr_fusion_competition_germany_test_labels/dlr_fusion_competition_germany_test_labels_33N_17E_243N/labels.geojson'

#s1 dirs 
s1_train = '../input/ai4eo-sentinel-1-tar/sentinel1_data/dlr_fusion_competition_germany_train_source_sentinel_1'
s1_test = '../input/ai4eo-sentinel-1-tar/sentinel1_data/dlr_fusion_competition_germany_train_source_sentinel_1'

#load test train labels
test_labels = gpd.read_file('./dlr_fusion_competition_germany_test_labels/dlr_fusion_competition_germany_test_labels_33N_17E_243N/labels.geojson')
train_labels = gpd.read_file('./dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson')

train_labels.head()
label_ids= train_labels['crop_id'].unique()
label_names= train_labels['crop_name'].unique()
Exploratory data Analysis
Helper functions

def plot_hist(arr,
              label,
              ax1=None,
              n_bins=30):
    
    if not ax1:
        ax1=plt.figure(figsize=(8,6))
        
    plt.hist(arr,bins=n_bins)
    plt.title(f'Histogram {label}')
    plt.ylabel('Count')
    plt.xlabel(f'{label}')
    
Distribution of crops

#set plt style
plt.style.use('Solarize_Light2')
plt.figure(figsize=(16,8))
sns.countplot(x=train_labels['crop_name'],orient='v')
plt.title('Crop instance Count')
Distribution of Area of Field

ax=plt.figure(figsize=(16,8))
plt.hist(train_labels['SHAPE_AREA'],bins=100)

print(f'mean Area of fields {train_labels.SHAPE_AREA.mean()}')
print(f'standard deviation Area of fields {train_labels.SHAPE_AREA.std()}')
print(f'median Area of fields {train_labels.SHAPE_AREA.median()}')
plt.title('Crop Area Distribution')
plt.subplots(3,3,figsize=(16,3*6))
for i,name in enumerate(train_labels['crop_name'].unique()):
    ax=plt.subplot(3,3,i+1)
    plot_hist(train_labels[train_labels['crop_name']==name]['SHAPE_AREA'],name+'_AREA',ax1=ax)
	plt.subplots(3,3,figsize=(16,3*6))
for i,name in enumerate(train_labels['crop_name'].unique()):
    ax=plt.subplot(3,3,i+1)
    plot_hist(train_labels[train_labels['crop_name']==name]['SHAPE_LEN'],name+'_LEN',ax1=ax)
Plot of fields.
Training data

#from https://github.com/AI4EO/tum-planet-radearth-ai4food-challenge/blob/main/notebook/starter-pack.ipynb


#DISPLAY TARGET FIELDS of 2018 FOR TRAINING ON THE MAP BY LABEL: 

#colors
colors_list = ['#78C850','#A8B820','#F8D030','#E0C068', '#F08030', '#C03028', '#F85888','#6890F0','#98D8D8'] 

fig, ax = plt.subplots(figsize=(18, 18))
counter=0
legend_elements = []
for name, group in train_labels.groupby('crop_name'):
    group.plot(ax=ax,color=colors_list[counter], aspect=1)
    legend_elements.append(Patch(facecolor=colors_list[counter], edgecolor=colors_list[counter],label=name))
    counter+=1

ax.legend(handles=legend_elements,loc='lower right')
ax.title.set_text('BRANDENBURG 2018: GROUND TRUTH POLYGONS WITH CROP LABELS for TRAINING')
#DISPLAY TARGET FIELDS of 2019 WITHOUT LABELS : 

fig, ax = plt.subplots(figsize=(18, 18))
counter=0
legend_elements = []
for name, group in test_labels.groupby('crop_name'):
    group.plot(ax=ax,color=colors_list[counter], aspect=1)
    legend_elements.append(Patch(facecolor=colors_list[counter], edgecolor=colors_list[counter],label=name))
    counter+=1

ax.legend(handles=legend_elements,loc='lower right')
ax.title.set_text('BRANDENBURG 2019: GROUND TRUTH POLYGONS WITHOUT CROP LABELS for the EVALUATION')
