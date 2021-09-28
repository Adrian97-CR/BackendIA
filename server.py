#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask,request, jsonify
from flask_cors import CORS
import pandas as pd # Librería de tratamiento de datos
import numpy as np  # Librería para operaciones matemáticas
from sklearn.ensemble import RandomForestClassifier
import pickle

from werkzeug.datastructures import ContentSecurityPolicy


app = Flask(__name__)
CORS(app)

# @app.route("/search_function", methods=['POST'])
# def search_func():
#     return (search_function(request.form['code'],request.form['search']))


@app.route("/avocadoPrice", methods=['POST'])
def get_AvocadoPrice():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
    #print(data)
    loaded_model = pickle.load(open("prediccion_precioAguacate", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result})

@app.route("/changeTp", methods=['POST'])
def get_ChangeTp():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorClientesPasanCompañia", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})

@app.route("/vehiclePrice", methods=['POST'])
def get_VehiclePrice():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
    #print(DF)
    loaded_model = pickle.load(open("prediccion_precioAutomovil", 'rb'))
    #print("lda")
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
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
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
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorDeCirrhosis", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/wineQuality", methods=['POST'])
def get_WineQuality():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4'], data['d5'], data['d6'], data['d7'], data['d8'], data['d9'], data['d10'], data['d11'], data['d12'], data['d13']]])
    #print(DF)
    loaded_model = pickle.load(open("modelo_clasificadorDeVino", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})
    
@app.route("/homeRental", methods=['POST'])
def get_HomeRental():
    data = request.json
    DF=pd.DataFrame([[data['d1'], data['d2'], data['d3'], data['d4']]])
    #print(DF)
    loaded_model = pickle.load(open("prediccion_precioCasas", 'rb'))
    #print("lda")
    result = loaded_model.predict(DF)
    return jsonify({"response":result[0]})


if __name__ == "__main__":
    app.run(host='0.0.0.0')









