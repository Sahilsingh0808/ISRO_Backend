from base64 import encode
from crypt import methods
import csv
import os
from typing import final
from unittest import result
from flask import Flask, flash, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
from ml import *
import json
from json import JSONEncoder
from pathlib import Path


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['.csv', '.xlsx', '.lc', '.ascii'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# members=["John", "Mary", "Peter"]

# #Members API Route
# @app.route("/members",methods=["GET"])
# def members():
#     return {"members": ["John", "Mary", "Peter"]}

b=[]
c=[]
d=[]

@app.route("/",methods=["GET"])
def hello_world():
    
    return "JuX Server"

@app.route("/members",methods=["POST"])
def membersPost():
    member=request.json['name']
    members.append(member)
    return jsonify({'members':members})

# @app.route('/upload', methods=['POST'])
# def fileUpload():
#     target=os.path.join(UPLOAD_FOLDER,'test_docs')
#     if not os.path.isdir(target):
#         os.mkdir(target)
#     logger.info("welcome to upload`")
#     file = request.files['file'] 
#     filename = secure_filename(file.filename)
#     destination="/".join([target, filename])
#     file.save(destination)
#     session['uploadFilePath']=destination
#     response="File Successfully Uploaded"
#     return response

def output(name):
    downloads_path = str(Path.home() / "Downloads")
    if not os.path.exists(downloads_path+"/uploads"):
        os.makedirs(downloads_path+"/uploads")
    path_to_lc=uploads_dir+"/"+name
    print("PATH")
    print(path_to_lc)
    print(downloads_path)
    # path_to_lc="/home/sahilsingh/Documents/Repositories/MP_ISRO_T10/ISRO__Backend/instance/uploads/"+name
    # rand_lc = lightcurve(path_to_lc, should_plot=False)
    # print(rand_lc)
    # b=np.array(rand_lc)
    # rand_lc = lightcurve(path_to_lc, should_plot=False)
    # #np.array(rand_lc) jsonify this
    # xnew, ynew = smoothening(rand_lc, 40)    
    # c=xnew
    # d=ynew
    # print(xnew,ynew)

    # _s0, _p0 = get_lvl_0_extremas(xnew, ynew, should_plot=False)
    # _s1, _p1 = get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False)
    # _s2, _p2 = get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False)
    # _s3, _p3 = get_lvl_3_extremas(xnew, ynew, _s2, _p2, should_plot=False)
    # _s4, _p4 = get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False)
    
    # _e0 = get_lvl_0_ends(xnew, ynew, _s4, _p4, _s2)
    # _e1 = get_lvl_1_ends(xnew, ynew, _s0, _p4, _e0)

    # _zip = get_interm_zip(ynew, _s4, _p4, _e1)
    # final_zip = get_final_zip(xnew, ynew, _zip)
    # #finalzip.tojson
    # print(final_zip)
    # return [final_zip,np.array(rand_lc),xnew,ynew]
    x_arr, y_arr = lightcurve(path_to_lc)
    x_new = []
    y_new = []
    for i in range(len(x_arr)):
        window_sz = 20 + 100 * int(1 / (1 + np.exp(-1 * (len(x_arr[i]) - 240))))
        if (len(x_arr[i]) >= 120):
            _x, _y = smoothening_ma(x_arr[i], y_arr[i], 2*window_sz, window_sz//2)
            for j in range(len(_x)):
                x_new.append(_x[j])
                y_new.append(_y[j])

    xnew = np.linspace(int(x_new[0]), int(x_new[-1]-x_new[0]), int(x_new[-1]-x_new[0]))
    f__ = scipy.interpolate.interp1d(x_new, y_new, fill_value='extrapolate', kind='linear')
    ynew = f__(xnew)
    _s0, _p0 = get_lvl_0_extremas(xnew, ynew, should_plot=False)
    _s1, _p1 = get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False)
    _s2, _p2 = get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False)
    _s3, _p3 = get_lvl_3_extremas(xnew, ynew, _s2, _p2, 0.3, should_plot=False)
    _s4, _p4 = get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False)
    _s5, _p5 = get_lvl_5_extremas(xnew, ynew, _s4, _p4, should_plot=False)

    _e0 = get_lvl_0_ends(xnew, ynew, _s5, _p5, _s0, should_plot=False)
    _e1 = get_lvl_1_ends(xnew, ynew, _s0, _p5, _e0, should_plot=False)
    if len(_e1) != 0:
        h1, h2, h3, h4 = get_interm_zip_features(ynew, _s5, _p5, _e1)
        if len(h1) != 0:
            _zip = get_interm_zip(h1, h2, h3, h4)
            g1, g2, g3, g4, g5, g6, g7, g8, g9 = get_final_zip_features(xnew, ynew, _zip)
            if len(g1) != 0:
                final_zip = get_final_zip(g1, g2, g3, g4, g5, g6, g7, g8, g9)
                model_zip = get_model_features(final_zip, path_to_lc)
                print(final_zip)
    return [final_zip,"1",xnew,ynew,x_new,y_new]

uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

def listString(a):
    stri=""
    for i in range(0,len(a)):
        stri=stri+str(a[i])+"%%"
    return str

@app.route('/api/upload', methods = ['POST','GET'])
def upload_file():
    file = request.files['file']
    print(file)
    print(type(file))
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    output_arr=output(file.filename)
    final_zip=output_arr[0]
    csvdf = final_zip.to_csv()
    # csvdf.columns.values[0]="Start Time"
    # csvdf.columns.values[0]="Peak Time"
    # csvdf.columns.values[0]="End Time"
    # csvdf.columns.values[0]="Est End Time"
    # csvdf.columns.values[0]="Peak Intensity"
    # csvdf.columns.values[0]="Error"
    return csvdf

@app.route('/api/graph', methods = ['POST','GET'])
def graph():
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    output_arr=output(file.filename)
    rand_lc=output_arr[1]
    encodedNumpyData = json.dumps(np.array(rand_lc), cls=NumpyArrayEncoder)  # use dump() to write array into file
    return encodedNumpyData

@app.route('/api/x', methods = ['POST','GET'])
def x():
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    output_arr=output(file.filename)
    x=output_arr[2]
    print(x)
    encodedNumpyData = json.dumps(x, cls=NumpyArrayEncoder)  # use dump() to write array into file
    return encodedNumpyData

@app.route('/api/y', methods = ['POST','GET'])
def y():
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    output_arr=output(file.filename)
    y=output_arr[3]
    print(y)
    encodedNumpyData = json.dumps(y, cls=NumpyArrayEncoder)  # use dump() to write array into file
    return encodedNumpyData

@app.route('/api/xOri', methods = ['POST','GET'])
def xOri():
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    output_arr=output(file.filename)
    x=output_arr[4]
    encodedNumpyData = json.dumps(x, cls=NumpyArrayEncoder)  # use dump() to write array into file
    return encodedNumpyData

@app.route('/api/yOri', methods = ['POST','GET'])
def yOri():
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    output_arr=output(file.filename)
    y=output_arr[5]
    encodedNumpyData = json.dumps(y, cls=NumpyArrayEncoder)  # use dump() to write array into file
    return encodedNumpyData

# @app.route('/api/pointColor',methods=['POST','GET'])
# def pointColor():
#     file = request.files['file']
#     file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
#     output_arr=output(file.filename)
#     final_zip=output_arr[0]
#     start_time=final_zip.iloc[:,0].tolist()
#     peak_time=final_zip.iloc[:,1].tolist()
#     end_time=final_zip.iloc[:,2].tolist()
#     x=output_arr[3]
#     pointColor=[]
#     print("HERE")
#     print(type(len(x)))
#     pointColor = np.empty(len(x), dtype = object)
#     try:
#         pointColor.fill("#2380f7")
#         print("FILLING",pointColor)
#         for i in range(0,len(start_time)):
#             for j in range(start_time[int(i)],end_time[int(i)]):
#                 pointColor[int(j)]="#99BEEE"
#             print("FILLING",pointColor)
#         for i in range(0,peak_time):
#             pointColor[int(i)]="#FFFFFF"
#     except:
#         print("ERROR")
#     print("FILLING",pointColor)
#     # if len(x)!=len(pointColor):
#     #     a=len(x)-len(pointColor)

#     encodedNumpyData = json.dumps(pointColor, cls=NumpyArrayEncoder)
#     return encodedNumpyData

# @app.route('/api/pointWidth',methods=['POST','GET'])
# def pointWidth():
#     file = request.files['file']
#     file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
#     output_arr=output(file.filename)
#     final_zip=output_arr[0]
#     peak_time=final_zip.iloc[1].tolist()
#     x=output_arr[3]
#     pointWidth = np.empty(len(x), dtype = str)
#     try:
#         pointWidth.fill(1)
#         for i in range(0,peak_time):
#             pointWidth[int(i)]=4
#     except:
#         print("ERROR")
#     encodedNumpyData = json.dumps(pointWidth, cls=NumpyArrayEncoder)
#     print(encodedNumpyData)
#     return encodedNumpyData
    
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__=="__main__":
    app.run(debug=True)
