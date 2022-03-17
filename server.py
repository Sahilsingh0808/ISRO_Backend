from crypt import methods
import os
from unittest import result
from flask import Flask, flash, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
from meh import *
import json
from json import JSONEncoder

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
    return "ISRO Backend"

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
    path_to_lc="/home/sahilsingh/Documents/Repositories/ISRO_PS/isro/Backend/instance/uploads/"+name
    rand_lc = lightcurve(path_to_lc, should_plot=False)
    print(rand_lc)
    b=np.array(rand_lc)
    rand_lc = lightcurve(path_to_lc, should_plot=False)
    #np.array(rand_lc) jsonify this
    xnew, ynew = smoothening(rand_lc, 40)    
    c=xnew
    d=ynew
    print(xnew,ynew)

    _s0, _p0 = get_lvl_0_extremas(xnew, ynew, should_plot=False)
    _s1, _p1 = get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False)
    _s2, _p2 = get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False)
    _s3, _p3 = get_lvl_3_extremas(xnew, ynew, _s2, _p2, should_plot=False)
    _s4, _p4 = get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False)
    
    _e0 = get_lvl_0_ends(xnew, ynew, _s4, _p4, _s2)
    _e1 = get_lvl_1_ends(xnew, ynew, _s0, _p4, _e0)

    _zip = get_interm_zip(ynew, _s4, _p4, _e1)
    final_zip = get_final_zip(xnew, ynew, _zip)
    #finalzip.tojson
    print(final_zip)
    return [final_zip,np.array(rand_lc),xnew,ynew]

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


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__=="__main__":
    app.run(debug=True)
