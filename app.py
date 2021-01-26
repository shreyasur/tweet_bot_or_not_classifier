from flask import Flask,render_template,request
import pickle
import numpy as np



app=Flask(__name__)

clf=pickle.load(open('model/model3.pkl','rb'))
ctv=pickle.load(open('model/ctv.pkl','rb'))



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['post'])
def predict():
    # recieve form data here
    content = request.form.get('content')

    X=np.array([content])
    print(X)
    X_ctv=ctv.transform(X)


    y_pred = clf.predict(X_ctv)
    print(y_pred)

    def prediction(value):
       if value=='1':
         return 'bot'
       else:
         return 'not bot'

    # FOR DISPLAYING THE RESULT
    return render_template('index.html', result=prediction(y_pred[0]))






if __name__=="__main__":
    # IF WE KEEP DEBUG=TRUE THEN THE CHANGES ARE AUTOMATICALLY REFLECTED IN THE WEBPAGE
    app.run(debug=True)