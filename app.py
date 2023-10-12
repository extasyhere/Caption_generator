from flask import Flask, render_template, redirect, request
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import keras.utils as image
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input


with open("model.json",'r') as file:
    model=model_from_json(file.read())
model.load_weights('model.h5')

model.make_predict_function()

with open('word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)
with open('idx_to_word.pkl', 'rb') as f:
    idx_to_word = pickle.load(f)
    
with open("feature_model.json",'r') as file:
    feature_model=model_from_json(file.read())
feature_model.load_weights('feature_model.h5')

feature_model.make_predict_function()

max_len=31

def predict_caption(photo):
    in_text="start"
    for i in range(max_len):
        sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence=pad_sequences([sequence],maxlen=max_len,padding='post')
        ypred=model.predict([photo,sequence])
        ypred=ypred.argmax() #word with max probabilty -Greedy Sampling
        word=idx_to_word[ypred]
        in_text+=(" "+word)
        if word =='end':
            break
    final_caption=in_text.split()[1:-1]
    final_caption=' '.join(final_caption)
    return final_caption

#Extract image feature vectors
def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0) 
    #Normalisation
    img=preprocess_input(img)  #for resnet models only
    return img

def encode_image(img):
    img=preprocess_img(img)
    feature_vector=feature_model.predict(img)
    feature_vector=feature_vector.reshape((-1,))
    #print(feature_vector.shape)
    #it returns the feature vectors
    return feature_vector

def caption_this_image(image):
    feature_vector=encode_image(image)
       
    photo=feature_vector.reshape((1,2048))
    caption=str(predict_caption(photo))
    return caption

#__name ==__name__
app=Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def submit():
   if request.method=='POST':
       f=request.files['file']
       path="./static/{}".format(f.filename)
       f.save(path)
       caption=caption_this_image(path)
       #print(caption)
       result_dic={
           'image':path,
           'caption':caption
       }
       
   return render_template("index.html", result=result_dic)


if __name__=='__main__':
    app.debug=True #we dont need to restart the server every time we make a small change
    app.run()