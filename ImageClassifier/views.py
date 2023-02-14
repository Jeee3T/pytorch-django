from django.shortcuts import render
import torch
from PIL import Image
import numpy as np
import os 
from django.core.files.storage import FileSystemStorage
import pickle


media = 'media'
model= torch.load('best_model.pth')



def makepredictions(path):

    img= Image.open(path)
    img_d=img.resize((244,244))

    if len(np.array(img_d).shape)<4:
        rgb_img = Image.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img=img_d

    

    rgb_img=np.array(rgb_img, dtype=np.float64)
    rgb_img=rgb_img.reshape(1,244,244,3)

# predictions

    predictions=model.predict(rgb_img)
    a=int(np.argmax(predictions))
    if a==1:
        a="Result: ants"
    else: 
        a="Result: bees"
    return a




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
        predictions= makepredictions(os.path.join(media,file))
        return render(request,'index.html',{'pred':predictions,'file_url':file_url})



    return render(request,'index.html')
