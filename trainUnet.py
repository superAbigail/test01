#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import *
from data import *


# ## Train your Unet with membrane data
# membrane data is in folder membrane/, it is a binary classification task.
# 
# The input shape of image and mask are the same :(batch_size,rows,cols,channel = 1)

# ### Train with data generator

# In[2]:


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'D:/scau//U-net/unet-master/data/membrane/train','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])


# ### Train with npy file

# In[ ]:


#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])


# ### test your model and save predicted results

# In[3]:


testGene = testGenerator("D:/scau//U-net/unet-master/data/membrane/test")
model = unet()
model.load_weights("unet_membrane.hdf5")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("D:/scau//U-net/unet-master/data/membrane/test",results)

