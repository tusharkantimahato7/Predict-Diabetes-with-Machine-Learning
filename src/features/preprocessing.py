from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_features_target(data, target_column='Outcome'):
    """Split dataframe into features (X) and target (y)."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def split_train_test(X, y, test_size=0.3, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler