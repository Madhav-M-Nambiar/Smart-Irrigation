import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
df = pd.read_csv(r"C:\Users\m8494\Downloads\irrigation_machine.csv")
df.head()
df.info()
df.columns
df = df.drop('Unnamed: 0', axis=1)
df.head()
df.describe()
X = df.iloc[:, 0:20]   
y = df.iloc[:, 20:]
X.sample(10)
y.sample(10)
X.info()
y.info()
X
X.shape, y.shape
