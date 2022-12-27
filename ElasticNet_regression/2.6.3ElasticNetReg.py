import pandas as pd
from sklearn.datasets import load_boston

df=load_boston()

data=pd.DataFrame(df.data,columns=df.feature_names)

veri=data.copy()

veri["PRICE"]=df.target

y=veri["PRICE"]
X=veri.drop(columns="PRICE",axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import Ridge,Lasso,ElasticNet

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)

elas_model=ElasticNet(alpha=0.1)
elas_model.fit(X_train,y_train)

#print(ridge_model.score(X_train,y_train))
#print(lasso_model.score(X_train,y_train))
print(elas_model.score(X_train,y_train))

#print(ridge_model.score(X_test,y_test))
#print(lasso_model.score(X_test,y_test))
print(elas_model.score(X_test,y_test))

import sklearn.metrics as mt

#tahminrid=ridge_model.predict(X_test)
#tahminlasso=lasso_model.predict(X_test)
tahminelas=elas_model.predict(X_test)

#print(mt.mean_squared_error(y_test,tahminrid))
#print(mt.mean_squared_error(y_test,tahminlasso))
print(mt.mean_squared_error(y_test,tahminelas))

#Çarpraz doğrulama cross validation elasticnet reg de nasıl yapılır

from sklearn.linear_model import ElasticNetCV

lamb = ElasticNetCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_

elas_model2=ElasticNet(alpha=lamb)
elas_model2.fit(X_train,y_train)

print(elas_model2.score(X_train,y_train))
print(elas_model2.score(X_test,y_test))

tahminelas2=elas_model2.predict(X_test)
print(mt.mean_squared_error(y_test,tahminelas2))




