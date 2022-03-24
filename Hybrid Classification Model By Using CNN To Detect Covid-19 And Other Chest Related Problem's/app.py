import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['Covid','Normal','Others']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

        new_model = load_model('bestmodelresnet50fc1.h5')
        #new_model.summary()
        test_image = image.load_img('images\\'+filename,target_size=(299,299))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = new_model.predict(test_image)
        result1 = result[0]
        for i in range(3):
    
            if result1[i] == 1.:
                break;
        prediction = classes[i]
        if(prediction=='Others'):
            new_model1 = load_model('bestmodelmobilenetfc2.h5')
            result = new_model1.predict(test_image)
            result1 = result[0]
            for i in range(8):

                if result1[i] == 1.:
                    break;
            classes1 = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax']
            prediction = classes1[i]


    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("template.html",image_name=filename, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    #app.run(debug=False)
    app.run(host='127.0.0.1', port=56000, debug=True)

