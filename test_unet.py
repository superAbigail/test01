from model import *
from data import *
import os
from grad_cam import *
from PIL import Image

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model = unet()

testGene = testGenerator("/unet-master/test")


# imge = Image.open('D:/scau/U-Net/unet-master/test/3.png')
imge = load_image('/unet-master/test/3.png')
im = np.array(imge)
preditions = model.predict(imge)
top_1 = decode_predictions(preditions)[0][0]
predicted_class = np.argmax(preditions)

results = model.predict_generator(testGene, 30, verbose=1)

flag_multi_class = False
num_class = 2
item = results[4]
img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
io.imsave(os.path.join("/unet-master/test","aaaaa.png"),img)
# saveResult("D:/scau/U-Net/unet-master/test", results)

cam, heatmap = grad_cam(model, imge, results, "conv2d_17")
cv2.imwrite(os.path.join("/unet-master/test","a.png"), cam)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model)
saliency = saliency_fn([im, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
cv2.imwrite(os.path.join("/unet-master/test","aa.png"), deprocess_image(gradcam))