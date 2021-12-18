import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

raw = pd.read_csv('./data/data-head.csv')
obj_only = raw.select_dtypes('object')

s = (raw.dtypes == 'object')
object_cols = list(s[s].index)

ohe = OneHotEncoder(handle_unknown='ignore', drop='if_binary')
ohe.fit(obj_only)

encoded = ohe.transform(obj_only).toarray()
feature_names = ohe.get_feature_names(object_cols)

data = pd.concat([raw.select_dtypes(exclude='object'), 
                  pd.DataFrame(encoded,columns=feature_names).astype(int)], axis=1)
print(data)