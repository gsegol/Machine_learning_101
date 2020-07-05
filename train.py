# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-dir', required=True, help="Path to image directory")
ap.add_argument('-dict', required=False, default='cat_to_name.json', help="dictionary of objects to classify given as 'filename.json'")
ap.add_argument('-arch', required=False, default="vgg11", help=" pretrained model architecture: vgg11 (default) or resnet50")
ap.add_argument('-ep', required=False, default=3, help="number of iterations for training;default=3")
ap.add_argument('-hidden', required=False, default=512, help="number for hidden units; default=512")
ap.add_argument('-out', required=True, help="number of output units")
args = vars(ap.parse_args())

print(args)
#load the data
#data_dir = 'flowers'
print("this is the image directory: {}".format(args['dir']))
print("\n NOTE: It is assumed that your data are divided into three subdirectories: /train, /valid, /test for training, validation and testing, respectively")
data_dir = args['dir']
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print(train_dir)

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=train_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

#label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# # Building and training the classifier
# Build and train your network
if args['arch'] =='vgg11':
	model = models.vgg11(pretrained=True)
else:
	model = models.resnet50(pretrained=True)
model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#freeze parameters.  No backpropagation
for param in model.parameters():
    param.requires_grad = False

h_in = 25088 # using vgg11
h_hidden = int(args['hidden'])
h_out = int(args['out'])
print(h_in, h_hidden, h_out)
'''
if args['arch'] == resnet50:
	h_in =2048 #using resnet50  
'''
classifier = nn.Sequential(nn.Linear(h_in, h_hidden),
                          nn.ReLU(),
                           nn.Dropout(p=0.2,),
                          nn.Linear(h_hidden,h_out),
                          nn.LogSoftmax(dim=1))
model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

# ## Testing your network
# Do validation on the test set
train_losses, valid_losses = [], []
epochs = int(args['ep'])
print("\n epochs", epochs)
steps = 0

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad()
    
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
    
        running_loss +=loss.item()
    
    else:
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                log_ps = model.forward(inputs)
                valid_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                topcl = top_class ==labels.view(*top_class.shape)
                accuracy +=torch.mean(topcl.type(torch.FloatTensor))

            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))
            
            print ("Training loss: {:.3f}..".format(running_loss/len(trainloader)),
                  "Validation loss: {:.3f}..".format(valid_loss/len(validloader)),
                  "Validation accuracy:{:.3f}..".format(accuracy/len(validloader)))

    running_loss = 0
    model.train()

    # test
model.eval()
test_losses = []
test_loss = 0
accuracy = 0
print("\n starting validation test")
        
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        log_ps = model.forward(inputs)
        test_loss += criterion(log_ps, labels)
                
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        topcl = top_class ==labels.view(*top_class.shape)
        accuracy +=torch.mean(topcl.type(torch.FloatTensor))

        #train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
            
    print ("Test loss: {:.3f}..".format(test_loss/len(testloader)),
            "Test accuracy:{:.3f}..".format(accuracy/len(testloader)))

# ## Save the checkpoint
#print the network structure:
print("Dictionary: \n", model.state_dict().keys())
print("my model: \n\n", model)
model.class_to_idx = train_data.class_to_idx


checkpoint = {'input': h_in,
             'output': h_out,
             'hidden': h_hidden,
              'dropout': 0.2,
              'class_to_idx':model.class_to_idx,
             'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')

