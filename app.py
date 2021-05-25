from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import os

model=load_model('cat_dog.h5')

print('model loading successfull ')
print('Starting App')

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)                         # craete an application
app.config['UPLOAD_FOLDER'] = "data/"
# print("app_config:",app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')               
                                                       # POST is used to send data to a server to create/update a resource.
@app.route('/predict',methods=['POST'])          # GET is used to request data from a specified resource.
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    f = request.files['file'] 
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    print(f)    
    img=image.load_img(os.path.join(app.config['UPLOAD_FOLDER'],f.filename),target_size=(299,299))
    x = image.img_to_array(img)
    x = preprocess_input(x)                       #  this method to normalize the input data.The preprocess_input function is meant to adequate your image to the format the model requires
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]

    if pred[0]>0.5:
    	prediction='Yes its a cat!'
    else :
    	prediction='Nah! looks like a dog :/'

    return render_template('index.html', prediction_text='Answer : {}'.format(prediction))  # prediction_text : is a variable passing to html.

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host = "0.0.0",debug=True)    # host = "0.0.0" :now our app will not run on local host.                  


