import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
!pip install gdown
!gdown "https://drive.google.com/uc?id=1-EcjB6mHCWkYdeCpg9PDmEed_CHwIohJ"
import pickle
inception_model = pickle.load(open('inception_model.dt','rb'))
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

model_name = "inception"
num_classes = 30
batch_size = 256
num_epochs = 1
feature_extract = True
# device = torch.device("cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract = True, use_pretrained=True):
    model_ft = inception_model
    if model_name == "inception":
        #model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
print(model_ft)

#Define Dataset

from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage.color import gray2rgb
import skimage


class FacialDataset(Dataset):
    def __init__(self, filename):
        train_df = pd.read_csv(filename)
        self.x = train_df['Image'].to_numpy()
        train_df.drop(['Image'], axis=1, inplace=True)
        self.y = train_df.to_numpy()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        y = torch.Tensor(self.y[index])
        y = y.type(torch.cuda.FloatTensor)
        x_pixels = np.array(self.x[index].split(' '), dtype = float).reshape(96,96)
        x_resized = skimage.transform.resize(x_pixels,(299,299))
        x = torch.Tensor(gray2rgb(x_resized)).permute(2,0,1)/255     
        x = x.type(torch.cuda.FloatTensor)
        return x,y
        
#     pixels = np.array(x[i].split(' '), dtype = float).reshape(96,96)
#     x_resized[i] = skimage.transform.resize(pixels,(299,299))
train_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip')
x = train_df['Image']
x.to_numpy()
train_df.drop(['Image'], axis=1, inplace=True)
y = train_df.to_numpy()
torch.Tensor(y[0])
x_pixels = np.array(x[0].split(' '), dtype = float).reshape(96,96)
x_resized = skimage.transform.resize(x_pixels,(299,299))
x_resized.shape
x_tensor = torch.Tensor(gray2rgb(x_resized))
print(x_tensor.shape)
y = x_tensor.permute(2,0,1)/255
y.shape
train_df = pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip')
#image  = train_df.iloc[0]['Image']
train_numpy = train_df.to_numpy()
train_numpy[0]
y = train_df.iloc[:,0:29].to_numpy()
x = train_df.iloc[:,30].to_numpy()
#type(x)
#x.to_numpy()[0]
#train_df.shape()
pixels = np.array(x[0].split(' '), dtype = float).reshape(96,96)
pixels
#pixels.resize()
import skimage
from skimage.transform import resize  
#from skimage.color import rgb2gray
#gray = rgb2gray(pixels[0])
xnew = skimage.transform.resize(pixels,(299,299))
xnew.shape
type(xnew)
import skimage
from skimage.transform import resize  
n_rows = len(x)
x_resized = np.zeros((n_rows), dtype = np.ndarray )
#print(x_resized.shape)
for i in range(n_rows):
    pixels = np.array(x[i].split(' '), dtype = float).reshape(96,96)
    x_resized[i] = skimage.transform.resize(pixels,(299,299))
x_resized.dtype
# x_resized[0]
import matplotlib.pyplot as plt
plt.imshow(x_resized[2])
#plt.imshow(pixels)
x_resized[0].dtype
Plot image

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


# #plt.imshow(np.array(train_numpy[0][30]))
# pixels =np.array(image.split(' '), dtype =int).reshape(96,96)
# pixels
# plt.imshow(pixels)
#Define Dataloader

train_file = "/kaggle/input/facial-keypoints-detection/training.zip"
train_dataset = FacialDataset(train_file)
train_dl = DataLoader(train_dataset, batch_size, shuffle = True)
# for x,y in train_dl:
#     print(x,y)
len(train_dl.dataset)
Train model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, is_inception=True):
    since = time.time()
    best_acc = 0.0
    loss_hist = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        model.train()
        model.to("cuda")
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders:
            optimizer.zero_grad()
            outputs, aux_outputs = model(inputs)
            label_mask = (labels != labels)
            if label_mask.any():
                labels[label_mask] = outputs[label_mask]
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print('loss: {:.4f}'.format(loss))
            loss_hist.append(loss)
            

        epoch_loss = running_loss / len(dataloaders.dataset)
        print('Epoch {} Loss: {:.4f}'.format(epoch+1, epoch_loss))
        torch.save(model.state_dict(),'/kaggle/working/weights.dt')
        torch.save(loss, '/kaggle/working/loss.dt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, loss_hist
    
##Define Loss criterion and optimizer

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.MultiLabelSoftMarginLoss()    #for multilabel classification
model_ft.load_state_dict(torch.load('/kaggle/working/weights.dt'))
# Train and evaluate
model_ft, hist = train_model(model_ft, train_dl, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
model_ft.eval()
# train_file = "/kaggle/input/facial-keypoints-detection/training.zip"
# train_dl_test = DataLoader(FacialDataset(train_file), 1, shuffle = True)
for x, y in train_dl:
    y_pred = model_ft(x)
    print('Expected', y)
    print('Predicted', y_pred)
    break
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# #plt.imshow(np.array(train_numpy[0][30]))
# pixels =np.array(image.split(' '), dtype =int).reshape(96,96)
# pixels
# x = x.type(torch.cpu.FloatTensor)
# plt.imshow(x.argmax())
# x.argmax().numpy()

%matplotlib inline
#plt.figure(figsize=(2, 2))
RANGE = 20
COLS = 5
nrows = RANGE // COLS + 1
nrows
i = 73
x = train_dataset.x[i]
x_pixels = np.array(x.split(' '), dtype = float).reshape(96,96)
plt.imshow(x_pixels)
y_orig = train_dataset.y[i]
print(type(x))
y_orig = y_orig.reshape(-1,2)
x_tensor, _ = train_dataset.__getitem__(i)
print(type(x_tensor))
model_ft.eval()
y_pred = model_ft(x_tensor.unsqueeze(0))
print(y_orig)

y_pred = y_pred.cpu().detach().numpy().reshape(-1,2)
print(y_pred)
plt.scatter(y_orig[:, 0], y_orig[:, 1], marker = 'v', c = 'g')
plt.scatter(y_pred[:, 0], y_pred[:, 1], marker = '.', c = 'r')
%matplotlib inline
plt.figure(figsize = (15,10))
no_imgs = 6
no_cols = 5
no_rows = no_imgs//no_cols +1 
for i in range(no_imgs):
    x = train_dataset.x[i]
    x_pixels = np.array(x.split(' '), dtype = float).reshape(96,96)
    y_orig = train_dataset.y[i]
    y_orig = y_orig.reshape(-1,2)
    x_tensor, _ = train_dataset.__getitem__(i)
    model_ft.eval()
    y_pred = model_ft(x_tensor.unsqueeze(0))
    y_pred = y_pred.cpu().detach().numpy().reshape(-1,2)
    plt.subplot(no_rows, no_cols, i+1)
    plt.imshow(x_pixels)
    plt.title(f"Image {i+1}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.scatter(y_orig[:, 0], y_orig[:, 1], marker = 'v', c = 'g')
    plt.scatter(y_pred[:, 0], y_pred[:, 1], marker = '.', c = 'r')
id_table = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
test_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')

feature_map = {'left_eye_center_x': 0, 'left_eye_center_y': 1, 'right_eye_center_x' : 2,
       'right_eye_center_y' : 3, 'left_eye_inner_corner_x' : 4,
       'left_eye_inner_corner_y' : 5, 'left_eye_outer_corner_x' : 6,
       'left_eye_outer_corner_y' : 7, 'right_eye_inner_corner_x' : 8,
       'right_eye_inner_corner_y' : 9, 'right_eye_outer_corner_x' : 10,
       'right_eye_outer_corner_y' : 11, 'left_eyebrow_inner_end_x' : 12,
       'left_eyebrow_inner_end_y' : 13, 'left_eyebrow_outer_end_x' : 14,
       'left_eyebrow_outer_end_y' : 15, 'right_eyebrow_inner_end_x' : 16,
       'right_eyebrow_inner_end_y' : 17, 'right_eyebrow_outer_end_x' : 18,
       'right_eyebrow_outer_end_y' : 19, 'nose_tip_x' : 20, 'nose_tip_y' : 21,
       'mouth_left_corner_x' : 22, 'mouth_left_corner_y' : 23,
       'mouth_right_corner_x' : 24, 'mouth_right_corner_y' : 25,
       'mouth_center_top_lip_x' : 26, 'mouth_center_top_lip_y' : 27,
       'mouth_center_bottom_lip_x' : 28, 'mouth_center_bottom_lip_y' : 29
       }
model_ft.eval()
#img_data = test_data['Image'].to_numpy()
old_img_id = -1
for i in range(len(id_table)):
    new_img_id = id_table.loc[i,'ImageId']
    print(i)
    if (i == 0) or (new_img_id != old_img_id) :
        x = test_data.loc[new_img_id-1,'Image']
        img = np.array(x.split(' '), dtype = float).reshape(96,96)
        img = torch.Tensor(gray2rgb(skimage.transform.resize(img,(299,299)))).permute(2,0,1)/255
        img = img.type(torch.cuda.FloatTensor)
        features = model_ft(img.unsqueeze(0))
        features = features.cpu().detach().numpy().reshape(-1)
        old_img_id = new_img_id
    id_table.loc[i,'Location'] = features[feature_map[id_table.loc[i,'FeatureName']]]
    print(id_table.loc[i,'Location'])
    
id_table.drop(['ImageId'], axis=1, inplace=True)
id_table.drop(['FeatureName'], axis=1, inplace=True)
id_table = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
test_data = pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')

id_table.head(10).to_csv('/kaggle/working/IdLookupTable.csv', index = False)
test_data.head(10).to_csv('/kaggle/working/test.csv', index = False)
x = pd.read_csv('/kaggle/working/test.csv')
x
id_table.to_csv('/kaggle/working/saveIdLookupTable.csv', index = False )
x = pd.read_csv('/kaggle/working/saveIdLookupTable.csv')
x.head()
id_table_read = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
id_table_read.head()
#id_table.iloc[:, 2].head(31).to_numpy()
test_data.head(1)

# for img in test_data:
#     print(img)
#     break
img = test_data.loc[1,'Image']
print("imgae",img)

img = np.array(img.split(' '), dtype = float).reshape(96,96)
# img = gray2rgb(skimage.transform.resize(img(299,299)))
#img = img.type(torch.cuda.FloatTensor)
type(img)
img_data = test_data['Image'].to_numpy()
model_ft.eval()
img = np.array(img_data[0].split(' '), dtype = float).reshape(96,96)
img = gray2rgb(skimage.transform.resize(img,(299,299)))
img = torch.Tensor(img).permute(2,0,1)/255
img = img.type(torch.cuda.FloatTensor)
y = model_ft(img.unsqueeze(0))
y = y.cpu().detach().numpy().reshape(-1)
y[0]
x = feature_map[id_table.loc[1,'FeatureName']]
x
id_table.loc[0,'Location'] = 1
id_table.loc[0,'Location']
len(id_table)
test_data.loc[0,'ImageId']

