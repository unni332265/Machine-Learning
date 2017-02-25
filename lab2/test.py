import pandas as pd 
import numpy as np 

df = pd.read_csv('winequality-white.csv')
df['avg_acidity']= df['fixed acidity'] + df['volatile acidity']
print df 