import numpy as np


def train_test_split(data, target, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    
    # Random shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = data[train_indices]
    X_test = data[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]
    
    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def classification_report(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 60)


class LogisticRegression:
    """Logistic Regression implemented with NumPy.
    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Relative tolerance for early stopping based on loss improvement.
    """

    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.history = {"loss": [], "accuracy": []}

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        z = X @ self.weights
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def _binary_cross_entropy(self, y_true, y_pred_proba):
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        n = len(y_true)
        loss = -(1 / n) * np.sum(
            y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba)
        )
        return loss

    def fit(self, X, y, verbose=True):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        prev_loss = float("inf")

        for i in range(self.max_iters):
            z = X @ self.weights
            y_pred_proba = self._sigmoid(z)
            loss = self._binary_cross_entropy(y, y_pred_proba)
            gradient = (1 / n_samples) * X.T @ (y_pred_proba - y)
            self.weights -= self.learning_rate * gradient

            y_pred = (y_pred_proba >= 0.5).astype(int)
            acc = accuracy_score(y, y_pred)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)

            if abs(prev_loss - loss) < self.tol * prev_loss:
                if verbose:
                    print(f"Converged at iteration {i+1}")
                break
            prev_loss = loss

            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.max_iters}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        if verbose:
            print(f"Training completed. Final loss: {loss:.4f}, Final accuracy: {acc:.4f}")

    def evaluate(self, X, y, threshold=0.5, report=False):
        """Evaluate model on given data.

        Returns a dictionary of metrics. If report=True prints classification report.
        """
        y_pred = self.predict(X, threshold=threshold)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }
        if report:
            classification_report(y, y_pred)
        return metrics


