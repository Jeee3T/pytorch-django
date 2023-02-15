from django.shortcuts import render
import torch
from PIL import Image
import numpy as np
import os 
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
from itertools import chain

media = 'media'
model= torch.load('best_model.pth')

def makepredictions(path):

    img= Image.open(path) 
    # transform the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])


    image_transforms = transforms.Compose([
        transforms.Resize((224, 224,)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))

    ])

    img_d=image_transforms(img)
    img_d=img_d.unsqueeze(0)
    output=model(img_d)

    classes = [
    "ants",
    "bees",
    ]

    

    # calculating probability of one img with respect to the other image class
    acc = torch.nn.functional.softmax(output, dim=1)
    final_acc = acc * 100

    final_acc=final_acc.tolist() #convert into list

    # classes={0:["ants",final_acc[0]], 1:["bees",final_acc[1]] }

    _, predictions = torch.max(output.data, 1)
    res = classes[predictions.item()]
    #res=predictions.item()

    return final_acc
    
  



def index(request):

    if request.method == "POST" and request.FILES['upload']:

        if 'upload' not in request.FILES:
            err= 'No images selected'
            return render (request, 'index.html', {'err':err})
        f= request.FILES['upload']

        if f=='':
            err='No files selected'
            return render(request, 'index.html',{'err':err})
        
        upload= request.FILES['upload']
        fss= FileSystemStorage()
        file=fss.save(upload.name, upload)
        file_url=fss.url(file)
        predictions2d= makepredictions(os.path.join(media,file))
        #converting the 2d list into 1d
        predictions=[j for sub in predictions2d for j in sub]

        context={
            'ant':("%.2f" % predictions[0]),
            'bee': ("%.2f" % predictions[1]),
        }
         
        #key with maximum probability 
        keymax=max(zip(context.values(),context.keys()))[1]
        keymax=keymax.upper
                
        return render(request,'index.html',{'pred':context,'file_url':file_url,'max':keymax})



    return render(request,'index.html')
