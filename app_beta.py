
from google.cloud import vision
import os, io
import pandas as pd
from io import StringIO
import time
import codecs
import pythainlp
import cv2
from PIL import Image
from pythainlp import word_vector
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct

import streamlit as st

st.title('Welcome to My Ai Research App ❤️')
st.subheader('This is a simple app that can detect text in image and compare with your input✌️')
st.subheader('Please upload your image 📷')

# read file which contain authorized & enable google vision api
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'careful-flow-354414-83460c252874.json'

texts_data_gg_vision = []
# get text using google vision ai
emty_list = []
def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    #with io.open(path, 'rb') as image_file:
        #ontent = image_file.read()

    image = vision.Image(content=path)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    for text in texts:
        text_des = '\n"{}"'.format(text.description)
        emty_list.append(text_des)
    output = emty_list[0]
    texts_data_gg_vision.append(output)
    return output


#image path
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])
if uploaded_file is not None :
  uploaded_file_ = uploaded_file.read()

texts_data_gg_vision = []
emty_list = []
detect_text(uploaded_file_)
st.write(texts_data_gg_vision)
time.sleep(5)  

def tokenize(text):
  tokenizer = word_tokenize(text,engine="newmm")
  return tokenizer

def spell_cleaner(text):
  spell_clean = correct(text,engine = 'pn')
  return spell_clean

test_list = []
test_text = "ไขเจียวหมูสับ"
for i in range(len(texts_data_gg_vision)):
  token_new = tokenize(texts_data_gg_vision[i])
  #print(token_new)
  time.sleep(1)
  for j in range(len(token_new)):
    clean_word = spell_cleaner(token_new[j])
    #print(clean_word)
    test_list.append(clean_word)
#print(test_list)

labels =  word_vector.get_model().index_to_key

with st.form("my_form"):
  cant_eat = st.text_input("what are you allergic to","นม ไก่ ไข่")
  submitted = st.form_submit_button("Submit")
cant_eat= cant_eat.split()
list()

def cosine_sim_fuc(text_1, text_2):
  cosine_sim = word_vector.similarity(text_1, text_2)
  return cosine_sim

new_cant_eat = []
for i in range(len(cant_eat)):
  for j in range(len(test_list)):
    #detect_lag = Detector(test_list[j])
    #print(detect_lag)
    if test_list[j] in labels:
      sim_check = cosine_sim_fuc(cant_eat[i],test_list[j])
      time.sleep(1)
      #print(f"{cant_eat[i]},{test_list[j]},{sim_check:0.3f}")
      if sim_check >= 0.950:
        new_cant_eat.append(cant_eat[i])

#print(texts_data_gg_vision)

non_repeat_list = list(dict.fromkeys(new_cant_eat))
#print(non_repeat_list)
#print(f"ในสินค้ามีส่วนผสม {*non_repeat_list,}") 
if len(non_repeat_list) != 0:
  st.write(f"ในสินค้ามีส่วนผสม {*non_repeat_list,}")
else:
  st.write("ไม่มีส่วนผสมในสินค้านี้")
