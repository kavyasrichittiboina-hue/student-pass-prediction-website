import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template,request
app=Flask(__name__)
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
db_path=os.path.join(BASE_DIR,"stdprediction.db")

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
Y = np.array([0,0,0,0,1,1,1,1])
model=LogisticRegression()
model.fit(X,Y)
@app.route("/",methods=["GET","POST"])
def front():
    return render_template("front.html")

@app.route('/home',methods=['GET','POST'])
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
        probability=model.predict_proba(student)[0][1]
        conn=sqlite3.connect(db_path)
        cursor=conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS stdprediction(id INTEGER PRIMARY KEY AUTOINCREMENT,hours REAL,prediction INTEGER,probability REAL)""")
        cursor.execute("""INSERT INTO stdprediction(hours,prediction,probability) VALUES(?,?,?)""",(hours,int(prediction),float(probability)))
        conn.commit()
        print(os.path.abspath("stdprediction.db"))
        conn.close()
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

@app.route("/history")
def history():
    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()
    cursor.execute("SELECT * FROM stdprediction")
    data=cursor.fetchall()
    conn.close()
    return render_template("history.html",history=data)

@app.route("/delete/<int:id>",methods=["GET","POST"])
def delete(id):
    conn=sqlite3.connect(db_path)
    cursor=conn.cursor()
    if request.method=="POST":
        cursor.execute("DELETE FROM stdprediction where id=?",(id,))
        conn.commit()
        conn.close()
        return """<h1 style='color:black;  background-color:rgb(81, 104, 216); text-align:center; padding:20px;'> DELETED SUCCESSFULLY</h1>"""
    cursor.execute("SELECT * FROM stdprediction WHERE id=?",(id,))
    data=cursor.fetchone()
    conn.close()
    return render_template("delete.html",data=data)



if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)