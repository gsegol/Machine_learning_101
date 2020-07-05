from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import argparse

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-pic", required=True, help="path to the image to classify")
args = vars(ap.parse_args())

image_path = args['pic']
#image_path = 'flowers/test/10/image_07104.jpg'
#image_path = 'flowers/test/46/image_01078.jpg'
model = models.vgg11(pretrained=True)

def load_checkpoint(fpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn = torch.load(fpath, map_location=str(device))
    #fn = torch.load(fpath)
    classifier = nn.Sequential(nn.Linear(fn['input'], fn['hidden']),
                          nn.ReLU(),
                          nn.Dropout(p=fn['dropout']),
                          nn.Linear(fn['hidden'], fn['output']),
                          nn.LogSoftmax(dim=1))
   
    model.classifier = classifier
    model.load_state_dict(fn['state_dict'])
    #
    print("\n checkpoint data:")
    print(fn['input'])
    print(fn['output'])
    print(fn['hidden'])
    #dictionary
    #model.class_to_idx = fn['class_to_idx']
    class_to_idx = fn['class_to_idx']
    print(class_to_idx)
    
    return model, class_to_idx

model, class_to_idx = load_checkpoint('checkpoint.pth')
model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = torch.load('checkpoint.pth', map_location=str(device))

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
 # TODO: Process a PIL image for use in a PyTorch model
    #import numpy as np
    
    img = Image.open(image_path)
    print("original image size: ", img.size)
    
    w, h = img.size
    ar = w/h
    if ar>1:
        img = img.resize((int(256*ar), 256))
    else:
        img = img.resize((256, int(256*ar)))
    
    w, h = img.size
    l = (w - 224)/2
    r = (w + 224)/2
    t = (h - 224)/2
    b = (h + 224)/2
  
    img = img.crop((l,t,r,b))
    #print("processed image size", img.size)
  
    np_image = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_np_image = (np_image - mean)/std
    
    new_np_image_tr = new_np_image.transpose(2, 0, 1)
    
    proc_img = torch.from_numpy(new_np_image_tr).type(torch.FloatTensor)

    return proc_img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = "cpu"
    
    with torch.set_grad_enabled(False):
        #model.to(device)
        
        proc_img = process_image(image_path).unsqueeze_(0).type(torch.FloatTensor).to(device)
        
        output = model.forward(proc_img)            
        ps = torch.exp(output)
 
        top_p, top_class = ps.topk(topk, dim=1)
    
    return top_p.numpy().reshape(5), top_class.numpy().reshape(5)





#Invert the dictionary
idx_to_class = {}
for key, value in class_to_idx.items():
    idx_to_class[value] = key
print("Inverted dictionary \n", idx_to_class)

# ## Class Prediction

probs, classes = predict(image_path, model, topk=5)
print("Probabilities and top 5 classes")
print(probs)
print(classes)

#retrieve the class definitions
# loop through index to retrieve class from idx_to_class dict
top_class_loc = []
top_class_def =[]
for item in classes:
    top_class_loc.append(idx_to_class[item])
    top_class_def.append(cat_to_name[idx_to_class[item]])
print("\n Interpretation")
print(top_class_loc)
print(top_class_def)

# ## Sanity Check

def view_classify (ps, results):
    
    fig, ax2 = plt.subplots(figsize=(9,9))
    ax2.barh(np.arange(5), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(results, size='medium');
    ax2.set_title('Class Probability', size='medium')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    
proc_img = process_image(image_path)
img = imshow(proc_img)

view_classify(probs, top_class_def)


