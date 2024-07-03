
import numpy as np
import os
from prondict import prondict
from path import path_train
import os
from lab3_tools import *
from lab3_proto import *

from lab1_tools import *
from lab1_proto import *

from lab2_tools import *
from lab2_proto import *

traindata = np.load(path_train,allow_pickle=True)['traindata']



def get_indicies(index , last_index):
  left_index = index
  right_index = index
  t = [index]
  for i in range(3):

    left_index -=1
    t.insert(0,left_index)

    if(right_index==last_index):
       right_index =0
       t.append(right_index)
    else:
        right_index +=1
        t.append(right_index)

  for i in range(len(t)):
    if(t[i]<0):
      t[i] = -1 * t[i]

  return t





def Dynamic_Features(x):

  g = np.zeros((x.shape[0], 7* x.shape[1]))


  for i in range(x.shape[0]):
     list = get_indicies(i,x.shape[0]-1)
     temp = x[list ,:]
     g[i] = temp.flatten()

  return np.array(g)

data_train = []
data_validation =[]


#Creating train and validation data
#ensure that there are similar distribution between man and women

#6 men and 6 women to include in the validation set
#It will be 10 % of the entire data
id_validation = ['ae','aj','al','aw','bd','cb','ac','ag', 'ai', 'an','bh','bi']

counter_validation = 0
counter_train = 0
counter_all = 0

validation_c_man = 0
validation_c_women = 0

train_c_man = 0
train_c_women = 0
print(id_validation)
for i in range((len(traindata))):
  file_n = traindata[i]['filename']
  id = path2info(file_n)[1]
  gender = path2info(file_n)[0]
  

  N = traindata[i]['lmfcc'].shape[0]
  counter_all += N


  if id in id_validation:
   counter_validation  += N
   data_validation.append(traindata[i])
   if gender=='man':
    validation_c_man += N
   if gender=='woman':
    validation_c_women +=N

  else:
   data_train.append(traindata[i])
   counter_train +=N
   if gender=='man':
    train_c_man += N
   if gender=='woman':
    train_c_women +=N








print("Number of data: " + str(counter_all))

print("The percentage of validation_data: " + str(100*np.round(counter_validation/counter_all,2)))
print("The percentage of training_data:: " + str(100*np.round(counter_train/counter_all,2)))
print()
print("The persenatge of men in validation: "   + str(100*np.round(validation_c_man/counter_validation,2)))
print("The persenatge of women in validation: " + str(100*np.round(validation_c_women/counter_validation,2)))
print()
print("The percentage of men in the training_data: " + str(100*np.round(train_c_man/counter_train,2)))
print("The percentage of women in the training_data: " + str(100*np.round(train_c_women/counter_train,2)))



##Make single matrix for training data

all_train_lmfcc= []
all_train_mspec = []
all_train_targets = []
all_train_lmfcc_dynamic = []
all_train_mspec_dynamic = []

for i in range(len(data_train)):
  lmfcc = data_train[i]['lmfcc']
  mspec = data_train[i]['mspec']
  targets = data_train[i]['targets']

  all_train_lmfcc.append(lmfcc)
  all_train_mspec.append(mspec)

  all_train_targets.append(targets)

  all_train_lmfcc_dynamic.append(Dynamic_Features(lmfcc))
  all_train_mspec_dynamic.append(Dynamic_Features(mspec))

lmfcc_train_x = np.vstack(all_train_lmfcc)
mspec_train_x = np.vstack(all_train_mspec)
train_y = np.hstack(all_train_targets)
lmfcc_train_x_dynamic = np.vstack(all_train_lmfcc_dynamic)
mspec_train_x_dynamic = np.vstack(all_train_mspec_dynamic)

# np.save('lmfcc_train_x.npy', lmfcc_train_x)
# np.save('mspec_train_x.npy', mspec_train_x)
# np.save('train_y.npy', train_y)
#np.save('lmfcc_train_x_dynamic.npy', lmfcc_train_x_dynamic)
#p.save('mspec_train_x_dynamic.npy', mspec_train_x_dynamic)




#Make single matrix for validation data

all_validation_lmfcc= []
all_validation_mspec = []
all_validation_targets = []
all_validation_lmfcc_dynamic = []
all_validation_mspec_dynamic = []

for i in range(len(data_validation)):
  lmfcc = data_validation[i]['lmfcc']
  mspec = data_validation[i]['mspec']
  targets = data_validation[i]['targets']

  all_validation_lmfcc.append(lmfcc)
  all_validation_mspec.append(mspec)

  all_validation_targets.append(targets)

  all_validation_lmfcc_dynamic.append(Dynamic_Features(lmfcc))
  all_validation_mspec_dynamic.append(Dynamic_Features(mspec))

lmfcc_val_x = np.vstack(all_validation_lmfcc)
mspec_val_x = np.vstack(all_validation_mspec)
val_y = np.hstack(all_validation_targets)
lmfcc_validation_x_dynamic = np.vstack(all_validation_lmfcc_dynamic)
mspec_validation_x_dynamic = np.vstack(all_validation_mspec_dynamic)

print(mspec_validation_x_dynamic.shape)