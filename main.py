import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from flask import Flask,render_template,request
import numpy as np 

df=pd.read_csv("calories.csv")
#Data preprocessing
df=df.drop('User_ID',axis=1)
df['Gender']=df['Gender'].map({'male':0,'female':1})
#Train the Model
x=df.drop('Calories',axis=1)
y=df['Calories']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Evaluate the model
mae=mean_absolute_error(y_test,y_pred)

plt.scatter(y_test,y_pred)
plt.xlabel("actual calories")
plt.ylabel("predicted calories")
plt.title("actual vs predicted")

score=model.score(x_test,y_test)
# print("R2:",score)

coefficients=pd.DataFrame({'feature':x.columns,'coefficient':model.coef_})
# print(coefficients)

#Random_forest
rf_model=RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)
rf_pred=rf_model.predict(x_test)
mae=mean_absolute_error(y_test,rf_pred)
r2=r2_score(y_test,rf_pred)

#print("MAE:",mae)
#print("r2:",r2)

#importance or most impacted features
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_model.feature_importances_
})

#print(importance.sort_values(by='Importance', ascending=False))

#save the model in pickle
pickle.dump(rf_model,open('model.pkl','wb'))
#print("model saved")

#flask app
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@ app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    duration = float(request.form['duration'])
    heart_rate = float(request.form['heart_rate'])
    body_temp = float(request.form['body_temp'])

    features=pd.DataFrame([{'Gender':gender,'Age':age,'Height':height,'Weight':weight,'Duration':duration,'Heart_Rate':heart_rate,'Body_Temp':body_temp}])

    prediction =model.predict(features)

    return render_template('index.html',prediction_text=f'calories burned:{prediction[0]:.2f}')
    
if __name__=="__main__":
    app.run(debug=True)