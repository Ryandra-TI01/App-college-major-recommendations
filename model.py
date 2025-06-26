import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("dummy.csv")

    # Kolom kategori yang harus di-encode
    kategori = ['minat', 'karakter', 'gaya_belajar', 'suka_kerja_tim',
                'suka_tantangan', 'sekolah', 'lokasi', 'gaya_kerja', 'waktu_kerja', 'jurusan']
    
    encoders = {}
    for col in kategori:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].str.lower())
        encoders[col] = le

    X = df.drop(columns=['jurusan'])
    y = df['jurusan']
    return X, y, encoders, df

def train_model():
    X, y, encoders, _ = load_data()
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return model, encoders

def evaluate_model():
    X, y, _, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)
