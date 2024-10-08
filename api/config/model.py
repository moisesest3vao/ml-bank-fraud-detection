from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd
import config.config as config

clf = None
scaler = StandardScaler()
encoder = OneHotEncoder(drop='first', sparse_output=False)

def preprocess_features(X, scaler, encoder, categorical_columns, numerical_features):
    # Handle missing values
    if X.isna().any().any():
        X = X.fillna(X.median())
    
    # Encode categorical variables
    X_encoded = pd.DataFrame(encoder.transform(X[categorical_columns]), 
                             columns=encoder.get_feature_names_out(categorical_columns))
    X = X.drop(categorical_columns, axis=1).reset_index(drop=True)
    X = pd.concat([X, X_encoded], axis=1)
    
    # Scale numerical features
    X[numerical_features] = scaler.transform(X[numerical_features])
    
    return X

def initialize_model():
    global clf, scaler, encoder
    
    df = pd.read_csv(config.get_training_csv_path())
    
    X = df.drop(config.get_y(), axis=1)
    y = df[config.get_y()]
    
    if y.isna().any():
        raise ValueError("Target variable 'y' contains NaN values. Please clean the data before proceeding.")
    
    categorical_columns = config.get_categorical_columns()
    numerical_features = config.get_numerical_features()
    
    # Encode categorical variables
    encoder.fit(X[categorical_columns])
    
    # Scale numerical features
    scaler.fit(X[numerical_features])
    
    X = preprocess_features(X, scaler, encoder, categorical_columns, numerical_features)
    
    # Oversampling to handle class imbalance
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X, y = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
    print(f"F1 Scores from Cross-Validation: {scores}")
