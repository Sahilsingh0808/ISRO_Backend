from crypt import methods
import os
from flask import Flask, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['.csv', '.xlsx', '.lc', '.ascii'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

members=["John", "Mary", "Peter"]

#Members API Route
@app.route("/members",methods=["GET"])
def members():
    return {"members": ["John", "Mary", "Peter"]}

@app.route("/",methods=["GET"])
def hello_world():
    return "Hey"

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

@app.route('/api/upload', methods = ['POST'])
def upload_file():
    file = request.files['file']
    print(file)
    print(type(file))
    return "done"

if __name__=="__main__":
    app.run(debug=True)
