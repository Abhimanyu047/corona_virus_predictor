from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
import my_training
from sklearn.linear_model import LogisticRegression
import requests

app = Flask(__name__)

# load the model from pickle object
with open('model.pkl','rb') as f:
    clf = pickle.load(f)

@app.route("/",methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        print(f" form is {request.form}")
        form_dict = request.form
        fever = int(form_dict['fever'])
        age = int(form_dict['age'])
        bodyPain = int(form_dict['bodyPain'])
        runnyNose = int(form_dict['runnyNose'])
        diffBreath = int(form_dict['diffBreath'])

        print(f"selected values are: fever:{fever}, age: {age}, bodyPain: {bodyPain}, runnyNose: {runnyNose}, diffBreath{diffBreath}")
        # Code for inference
        input_features = [[fever,age,bodyPain,runnyNose,diffBreath]]
        infProb = clf.predict_proba(input_features)[0][1]
        return render_template('show.html', inf=infProb)
    
    return render_template('index.html' )


if __name__ == '__main__':
    # Read the data and train the model
    df = pd.read_csv('data.csv')
    train, test = my_training.data_split(df,0.2)
    X_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    X_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()

    Y_train = train['infectionProb'].to_numpy().reshape(2056 ,)
    Y_test = test['infectionProb'].to_numpy().reshape(513,)

    clf = LogisticRegression()
    clf.fit(X_train,Y_train)  

    # Saving the model (pickling) for future use
    with open('model.pkl','wb') as f:
        pickle.dump(clf, f)
    
    app.run(debug=True)