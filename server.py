#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask,request, jsonify
from flask_cors import CORS
import pandas as pd # Librería de tratamiento de datos
import numpy as np  # Librería para operaciones matemáticas
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from werkzeug.datastructures import ContentSecurityPolicy
#2nd part
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'C:/Users/adria/Documents/GitHub/IABack'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# @app.route("/search_function", methods=['POST'])
# def search_func():
#     return (search_function(request.form['code'],request.form['search']))
csv = ['background',
'vegetables | leafy_greens',
'vegetables | stem_vegetables',
'vegetables | non-starchy_roots',
'vegetables | other',
'fruits',
'protein | meat',
'protein | poultry',
'protein | seafood',
'protein | eggs',
'protein | beans/nuts',
'starches/grains | baked_goods',
'starches/grains | rice/grains/cereals',
'starches/grains | noodles/pasta',
'starches/grains | starchy_vegetables',
'starches/grains | other',
'soups/stews',
'herbs/spices',
'dairy',
'snacks',
'sweets/desserts',
'beverages',
'fats/oils/sauces',
'food_containers',
'dining_tools',
'other_food']

@app.route("/avocadoPrice", methods=['POST'])
def get_AvocadoPrice():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2']]])
    #print(DF)
    loaded_model = pickle.load(open("prediccion_precioAguacate", 'rb'))
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})

@app.route("/changeTp", methods=['POST'])
def get_ChangeTp():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4'], data['d5'], data['d6'], data['d7'], data['d8'], data['d7'], data['d8']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorClientesPasanCompañia", 'rb'))
    result = loaded_model.predict(DF)
    #print(result[0])
    return jsonify({"response":int(result[0])})

@app.route("/vehiclePrice", methods=['POST'])
def get_VehiclePrice():
    data = request.json
    DF=pd.DataFrame([[data['d1']]])
    loaded_model = pickle.load(open("prediccion_precioAutomovil", 'rb'))
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})

@app.route("/findIris", methods=['POST'])
def get_FindIris():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorEspeciesIris", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/rossmannCompany", methods=['POST'])
def get_RossmannCompany():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3']]])
    #print(DF)
    loaded_model = pickle.load(open("prediccion_ventasCompañiaRossmann", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/bodyFat", methods=['POST'])
def get_BodyFat():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2']]])
    #print(DF)
    loaded_model = pickle.load(open("prediccion_masaCorporal", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/hepatitisType", methods=['POST'])
def get_HepatitisType():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4'], data['d5'], data['d6'], data['d7'], data['d8'], data['d9'], data['d10'], data['d11'], data['d12']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorDeHepatitis", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/cirrosisType", methods=['POST'])
def get_CirrosisType():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4'], data['d5'], data['d6'], data['d7'], data['d8'], data['d9']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorDeCirrhosis", 'rb'))
    print("todo bien hasta aqui")
    result = loaded_model.predict(DF)
    print(result)
    return jsonify({"response":result[0]})
    
@app.route("/wineQuality", methods=['POST'])
def get_WineQuality():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4'], data['d5'], data['d6'], data['d7'], data['d8'], data['d9'], data['d10'], data['d11'], data['d12']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorDeVino", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/homeRental", methods=['POST'])
def get_HomeRental():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3']]])
    #print(DF)
    loaded_model = pickle.load(open("prediccion_precioCasas", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})


@app.route("/upload", methods=['POST'])
def upload():
    file = request.files['file']
    filename = "pexels.jpg"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image,dtype=tf.float16,saturate=False)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_with_pad(image,513,513)
    mpredictions = hub.KerasLayer('https://tfhub.dev/google/seefood/segmenter/mobile_food_segmenter_V1/1', output_key="food_group_segmenter:semantic_predictions")

    predictions = mpredictions(image)

    nuevo = tf.reshape(predictions,[513,513])
    result = tf.sparse.bincount(nuevo,minlength=1)
    vals =result.values.numpy()
    ind = result.indices.numpy()
    promVal = []
    relInd = []
    for i in range(1, vals.size):
        aux = round(vals[i]*100/(263169-vals[0]),0)
        if(1<aux):
            promVal.append(aux)
            relInd.append(csv[ind[i][0]])
    return jsonify({"response":[relInd, promVal]})


def contar(predictions):
  nuevo = tf.reshape(predictions,[513,513])
  outputs = tf.sparse.bincount(nuevo,axis=-1)
  return outputs
if __name__ == "__main__":
    app.run(host='0.0.0.0')




# def contar():
#   nuevo = tf.reshape(predictions,[513,513])
#   outputs = tf.sparse.bincount(nuevo,minlength=1)
#   return outputs

# #newpro = tf.cast(predictions,dtype=tf.float16)
# result = contar()

# vals =result.values.numpy()
# ind = result.indices.numpy()
# promVal = []
# relInd = []
# for i in range(1, vals.size):
#   aux = round(vals[i]*100/(263169-vals[0]),0)
#   if(1<aux):
#     promVal.append(aux)
#     relInd.append(csv[ind[i][0]])
#   #print(ind[i][0])
# print(promVal)
# print(relInd)
# ## Alli pueden sacar todos los 26 indices :)




