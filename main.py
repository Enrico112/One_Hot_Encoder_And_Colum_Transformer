import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# load df
df = pd.read_csv('homeprices.csv')

 # METHOD 1: dummy

# get dummy var for town col
dummies = pd.get_dummies(df.town)
# concatenate dfs along col axis
merged = pd.concat([df,dummies],axis='columns')

# dummy vars are multicollinear (one can be derived from other two)
# dropping one dummy and town cols
# thus final has only num cols and none is redundant
final = merged.drop(['town', 'west windsor'], axis='columns')

# create X df without price
X = final.drop('price', axis='columns')
# create y df with only price
y = final.price

# fit lin reg
model = LinearRegression()
# fit model
model.fit(X.values,y)

# predict 2800 sqft home in robinsville
model.predict([[2800,0,1]])
# predict 3400 sqft home in windsor
model.predict([[3400,0,0]])

# measure accuracy
model.score(X.values,y)

# METHOD 2: Column Transformer

# get Xs
X = df[['town','area']].values
# get y
y = df.price.values

# create CT obj, specify town is cat
ct = ColumnTransformer([("town", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

# drop 0th col because is dummy
X = X[:,1:]

# fit model
model.fit(X,y)

# predict 2800 sqft in robbinsville
model.predict([[1,0,2800]])
# predict 3400 sqft home in windsor
model.predict([[0,0,3400]])

# EXERCISE

df = pd.read_csv('carprices.csv')
X = df[['Car Model','Mileage','Age(yrs)']].values
y = df[['Sell Price($)']].values
ct = ColumnTransformer([("Car Model", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:,1:]
model.fit(X,y)
# Price of mercedez benz that is 4 yr old with mileage 45000
model.predict([[0,1,45000,4]])
# Price of BMW X5 that is 7 yr old with mileage 86000
model.predict([[1,0,86000,7]])
