from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_test, predictions):
    """Evaluate model performance with accuracy and classification report."""
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    return accuracy