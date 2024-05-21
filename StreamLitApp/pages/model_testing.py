import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import tempfile
from skimage import io
from skimage.transform import resize
from models import *
import mytransforms  as mytransforms
from models import *


st.set_page_config(
  page_title="Apeep",
  page_icon="H",
  layout="wide",
  initial_sidebar_state="expanded"
)
st.header("Model testing")

@st.cache_resource()
def get_model():
    file = open("resources/facial_exp.t7", "rb")
    print(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(file, map_location=device)


model = get_model()


with st.form("form"):

  st.subheader("Enter a sample to predict")
  uploaded_file = st.file_uploader("Upload you dataframe in csv")
  
  submitted = st.form_submit_button("Predict")

if submitted:
  if uploaded_file is not None:
    file_path = os.path.join(".", uploaded_file.name)
    with open(file_path,"wb") as f: 
      f.write(uploaded_file.getbuffer())         
    st.success("Saved File")
    print(file_path)

    cut_size = 44
    net = VGG('VGG19')

    transform_test = mytransforms.Compose([
        mytransforms.TenCrop(cut_size),
        mytransforms.Lambda(lambda crops: torch.stack([mytransforms.ToTensor()(crop) for crop in crops])),
    ])

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    raw_img = io.imread(file_path)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG19')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join('resources', 'facial_exp.t7'), map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)

    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes=plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()


    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    axes=plt.subplot(1, 3, 3)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    # show emojis

    #plt.show()
    plt.savefig(os.path.join('images/results/l.png'))
    plt.close()

    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

    st.write("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))
    st.image(os.path.join('images/results/l.png'))
"""
visualize results for test image
"""




