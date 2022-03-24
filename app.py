from flask import Flask,render_template ,request
import pickle
import numpy as np
mo = pickle.load(open('model.pkl','rb'))
#print(mo.predict([['645','2015']]))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = mo.predict(final_features)
    return render_template('index.html',prediction_text = 'predicted values is {}'.format(prediction[0]))
if __name__ == '__main__':
    app.run(debug=False)