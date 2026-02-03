from sklearn.linear_model import LogisticRegression

def create_model():
    """Create and return a logistic Regression model."""
    model = LogisticRegression(max_iter=200, random_state=42)
    return model
    