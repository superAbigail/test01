from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1, '/unet-master/data/membrane/train', 'image', 'label', data_gen_args, save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=50, epochs=20, callbacks=[model_checkpoint])

# testGene = testGenerator("D:/scau/U-Net/unet-master/data/membrane/test")
# results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("D:/scau/U-Net/unet-master/data/membrane/test", results)
testGene = testGenerator("/unet-master/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("/unet-master/test", results)