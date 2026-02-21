## Algorithm of choice
## Decision Tree Regression: basically goes through the data
# and finds the best feature and threshold to split the data to minimize the mean squared error in the child nodes.
# which creates a tree structure where each internal node represents a feature and threshold,
# and each leaf node represents a prediction (the mean of the target values in that leaf).

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data():
    """Load and preprocess the breast cancer dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize using training statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm, y_train, y_test, data.feature_names

def train(X_train, y_train, max_depth=5, min_samples_split=5):
    clf = DecisionTreeClassifier(max_depth= max_depth, random_state=42, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)
    return clf

# =============================================================================
# PART 4: Evaluation (provided)
# =============================================================================

def predict(clf, X):
    """Make predictions for multiple samples."""
    return clf.predict(X)

def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return accuracy_score(y_true, y_pred)


# =============================================================================
# PART 5: Main (run training and evaluation)
# =============================================================================

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")
    print("-"*40+"\nDecision Tree Classifier")
    # Train
    print("\nTraining...")
    clf = train(X_train, y_train, max_depth=5, min_samples_split=5)
    
    # Evaluate
    print("\nEvaluating...")
    train_pred = predict(clf, X_train)
    test_pred = predict(clf, X_test)
    
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("-"*40+"\nSelf-Made Binary Classifier")
    import binary_classification as bc
    X_train, X_test, y_train, y_test, feature_names = bc.load_data()
    print("\nTraining...")
    w, b, losses = bc.train(X_train, y_train, alpha=0.01, n_epochs=100)

    train_pred = bc.predict(X_train, w, b)
    test_pred = bc.predict(X_test, w, b)
    
    train_acc = bc.accuracy(y_train, train_pred)
    test_acc = bc.accuracy(y_test, test_pred)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

# reflection:
# the binary classifier achieved a higher test accuracy compared to the decision tree, 
# whereas the decision tree had a slightly higher training accuracy. I think the tree may have overfitted the data,
# while the binary classifier i made works better for this test set.
# 

