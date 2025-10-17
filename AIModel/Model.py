import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv(r"Forward_simulator/final_pts.csv")
data["p_f"] = (data["p_f"].apply(lambda x: list(map(float,x.strip("[]").split(',')))[0:2]))
p_f_coords = data['p_f'].apply(pd.Series)
p_f_coords = p_f_coords.rename(columns={0: 'land_x', 1: 'land_y'})
data = pd.concat([data, p_f_coords], axis=1)
data = data.drop('p_f', axis=1)

Velocity_ranges = [0,22,26,np.inf]
labels = [0,1,2]
data["v_cat"] = pd.cut(data['v_mag'],bins=Velocity_ranges,labels=labels,right=False)

features = ['land_x','land_y']
labels = ['p_x','p_y','p_z','v_mag','phi','w_y']

X = data[features]
y = data[labels]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train,X_test, y_train, y_test =train_test_split( X_scaled,y_scaled,test_size=0.2,random_state=42)

model = Sequential([

    Dense(64,activation='relu',input_shape = (3,)),

    Dense(128, activation='relu'),

    Dense(128, activation='relu'),

    Dense(128, activation='linear')

])

model.summary()

model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))

print(data)