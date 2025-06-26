import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("dataset.csv")
    le_minat = LabelEncoder()
    le_karakter = LabelEncoder()
    le_jurusan = LabelEncoder()

    df['minat'] = le_minat.fit_transform(df['minat'])
    df['karakter'] = le_karakter.fit_transform(df['karakter'])
    df['jurusan'] = le_jurusan.fit_transform(df['jurusan'])

    X = df[['matematika', 'ipa', 'ips', 'minat', 'karakter']]
    y = df['jurusan']

    return X, y, le_minat, le_karakter, le_jurusan, df


def train_model():
    X, y, le_minat, le_karakter, le_jurusan, _ = load_data()
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    return model, le_minat, le_karakter, le_jurusan

def evaluate_model():
    X, y, *_ = load_data()
    model = DecisionTreeClassifier(max_depth=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
