#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
y=X_train.SalePrice
cols_with_missing_1=['GarageYrBlt','MasVnrArea','SalePrice','SaleCondition','SaleType', 'PoolQC','Fence', 'PoolArea', 'ScreenPorch','3SsnPorch','PavedDrive', 'GarageCond', 'GarageQual','FireplaceQu', 'Functional', 'BsmtHalfBath', 'LowQualFinSF','Heating','BsmtCond','ExterCond','RoofMatl','Alley','LotFrontage']
cols_with_missing=['GarageYrBlt','MasVnrArea','SaleCondition','SaleType', 'PoolQC','Fence', 'PoolArea', 'ScreenPorch','3SsnPorch','PavedDrive', 'GarageCond', 'GarageQual','FireplaceQu', 'Functional', 'BsmtHalfBath', 'LowQualFinSF','Heating','BsmtCond','ExterCond','RoofMatl','Alley','LotFrontage']
Reduced_X_train=X_train.drop(cols_with_missing_1,axis=1)
Reduced_X_test=X_test.drop(cols_with_missing,axis=1)
s = (Reduced_X_train.dtypes == 'object')
object_cols = list(s[s].index)
categoric_train=Reduced_X_train.select_dtypes(exclude=[np.number])
categoric_test=Reduced_X_test.select_dtypes(exclude=[np.number])
numeric_train = Reduced_X_train.select_dtypes(include=[np.number])
numeric_test= Reduced_X_test.select_dtypes(include=[np.number])
from sklearn.impute import SimpleImputer
# Imputation for categorical data
my_imputer = SimpleImputer(strategy='most_frequent')
imputedc_X_train = pd.DataFrame(my_imputer.fit_transform(categoric_train))
imputedc_X_test = pd.DataFrame(my_imputer.transform(categoric_test))
# Imputation removed column names; put them back
imputedc_X_train.columns = categoric_train.columns
imputedc_X_test.columns = categoric_test.columns

#imputation for numerical data
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='mean')
imputedn_X_train = pd.DataFrame(my_imputer.fit_transform(numeric_train))
imputedn_X_test = pd.DataFrame(my_imputer.transform(numeric_test))
# Imputation removed column names; put them back
imputedn_X_train.columns = numeric_train.columns
imputedn_X_test.columns = numeric_test.columns

# Make copy to avoid changing original data for categorical data
label_X_train = imputedc_X_train.copy()
label_X_test = imputedc_X_test.copy()

# Apply label encoder to each column with categorical data
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(imputedc_X_train[col])
    label_X_test[col] = label_encoder.transform(imputedc_X_test[col])

result_train = pd.concat([imputedn_X_train,label_X_train],axis=1)
result_test = pd.concat([imputedn_X_test,label_X_test],axis=1)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(result_train,y)
Model_preds = forest_model.predict(result_test)

print(Model_preds)
prediction_file<-tibble(Id=result_train.Id,SalePrice=Model_preds)
write_csv(prediction_file,'submission.csv')


# In[52]:


result_train.info()
result_test.info()


# In[ ]:




