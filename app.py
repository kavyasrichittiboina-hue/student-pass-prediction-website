import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template,request
app=Flask(__name__)
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
Y = np.array([0,0,0,0,1,1,1,1])
model=LogisticRegression()
model.fit(X,Y)
@app.route('/',methods=['GET','POST'])
def home():
    prediction=None
    probability=None
    result=None
    if request.method=="POST":
        hours=float(request.form['hours'])
        student=[[hours]]
        prediction=model.predict(student)[0]
        if prediction==1:
            result="Pass"
        else:
            result="Fail"
        probability=model.predict_proba(student)
        x_values=np.linspace(0,10,100).reshape(-1,1)
        y_prob=model.predict_proba(x_values)[:,1]
        plt.scatter(X,Y,color='red',label="students")
        plt.plot(x_values,y_prob,color='blue',label="Pass probability")
        plt.axhline(y=0.5,color='green',linestyle='--',label="Threshold")
        plt.xlabel("Study Hours")
        plt.ylabel("Pass Probability")
        plt.title("Logistic Regression")
        plt.legend()
        graph_path=os.path.join(app.root_path,'static','logistic_regression.png')
        plt.savefig(graph_path)
        print("Graph saved successfully")
        plt.close()
    return render_template('home.html',result=result,prediction=prediction,probability=probability)
if __name__=="__main__":
    app.run(debug=True)