import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List
import joblib
import os


class OneClassSVMClassifier:
    """
    A wrapper class for one-class SVM with additional functionality for anomaly detection.
    
    This class provides a convenient interface for training, evaluating, and using
    one-class SVM models for outlier detection and novelty detection tasks.
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 nu: float = 0.1,
                 gamma: str = 'scale',
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize the OneClassSVM classifier.
        
        Args:
            kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            nu: An upper bound on the fraction of training errors and a lower bound
                of the fraction of support vectors. Should be in the interval (0, 1].
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
                  'scale' uses 1 / (n_features * X.var()) as value of gamma.
            random_state: Random state for reproducibility
            verbose: Enable verbose output
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize the SVM model
        self.svm = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            verbose=verbose
        )
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Model state
        self.is_fitted = False
        self.n_features_ = None
        self.n_samples_ = None
        self.support_vectors_ = None
        self.decision_function_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OneClassSVMClassifier':
        """
        Fit the one-class SVM model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API consistency
            
        Returns:
            self: The fitted model
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Store original dimensions
        self.n_samples_, self.n_features_ = X.shape
        
        # Fit the scaler and transform the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the SVM model
        self.svm.fit(X_scaled)
        
        # Store support vectors and decision function
        self.support_vectors_ = self.svm.support_vectors_
        self.decision_function_ = self.svm.decision_function(X_scaled)
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"Model fitted with {len(self.support_vectors_)} support vectors")
            print(f"Training data shape: {X.shape}")
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Samples to predict, shape (n_samples, n_features)
            
        Returns:
            Array of predictions: 1 for inliers, -1 for outliers
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.svm.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function of the samples.
        
        Args:
            X: Samples to evaluate, shape (n_samples, n_features)
            
        Returns:
            Array of decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing decision function")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Compute decision function
        return self.svm.decision_function(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the anomaly score for each sample.
        
        Args:
            X: Samples to score, shape (n_samples, n_features)
            
        Returns:
            Array of anomaly scores (higher values indicate more anomalous)
        """
        # For one-class SVM, the negative decision function gives anomaly scores
        return -self.decision_function(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates for samples.
        
        Note: One-class SVM doesn't provide true probabilities, so this returns
        normalized decision function values as a proxy.
        
        Args:
            X: Samples to predict, shape (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, 2) with [outlier_prob, inlier_prob]
        """
        decision_scores = self.decision_function(X)
        
        # Normalize to [0, 1] range
        scores_normalized = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-8)
        
        # Convert to probabilities: [outlier_prob, inlier_prob]
        outlier_prob = 1 - scores_normalized
        inlier_prob = scores_normalized
        
        return np.column_stack([outlier_prob, inlier_prob])
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate the model performance.
        
        Args:
            X: Test data, shape (n_samples, n_features)
            y_true: True labels, shape (n_samples,) where 1=inlier, -1=outlier
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        # Get predictions and scores
        y_pred = self.predict(X)
        scores = self.score_samples(X)
        
        # Convert labels to binary (0=outlier, 1=inlier)
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true_binary, scores)
        except ValueError:
            roc_auc = 0.5  # Default value when only one class is present
            
        # Precision-Recall AUC
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, scores)
            pr_auc = auc(recall_curve, precision_curve)
        except ValueError:
            pr_auc = 0.0
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def plot_decision_boundary(self, X: np.ndarray, y_true: Optional[np.ndarray] = None,
                              resolution: int = 100, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot the decision boundary for 2D data.
        
        Args:
            X: Input data, shape (n_samples, 2)
            y_true: True labels (optional)
            resolution: Resolution of the decision boundary grid
            figsize: Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        if X.shape[1] != 2:
            raise ValueError("Plotting is only supported for 2D data")
            
        # Create a grid of points
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # Get predictions for grid points
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points).reshape(xx.shape)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.contour(xx, yy, Z, colors='black', alpha=0.6, linewidths=0.5)
        
        # Plot data points
        if y_true is not None:
            inliers = X[y_true == 1]
            outliers = X[y_true == -1]
            
            if len(inliers) > 0:
                plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', 
                           label='Inliers', alpha=0.7, s=50)
            if len(outliers) > 0:
                plt.scatter(outliers[:, 0], outliers[:, 1], c='red', 
                           label='Outliers', alpha=0.7, s=50)
        else:
            plt.scatter(X[:, 0], X[:, 1], c='black', alpha=0.6, s=30)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('One-Class SVM Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_features_': self.n_features_,
            'n_samples_': self.n_samples_,
            'support_vectors_': self.support_vectors_,
            'decision_function_': self.decision_function_,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'OneClassSVMClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded OneClassSVMClassifier instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            kernel=model_data['kernel'],
            nu=model_data['nu'],
            gamma=model_data['gamma'],
            random_state=model_data['random_state'],
            verbose=model_data['verbose']
        )
        
        # Restore model state
        instance.svm = model_data['svm']
        instance.scaler = model_data['scaler']
        instance.n_features_ = model_data['n_features_']
        instance.n_samples_ = model_data['n_samples_']
        instance.support_vectors_ = model_data['support_vectors_']
        instance.decision_function_ = model_data['decision_function_']
        instance.is_fitted = model_data['is_fitted']
        
        return instance
    
    def get_params(self) -> dict:
        """Get the model parameters."""
        return {
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'OneClassSVMClassifier':
        """Set the model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Reinitialize SVM if parameters changed
        if any(key in ['kernel', 'nu', 'gamma', 'random_state'] for key in params):
            self.svm = OneClassSVM(
                kernel=self.kernel,
                nu=self.nu,
                gamma=self.gamma,
                verbose=self.verbose
            )
            self.is_fitted = False
            
        return self
    
    def __repr__(self):
        """String representation of the model."""
        return (f"OneClassSVMClassifier(kernel='{self.kernel}', nu={self.nu}, "
                f"gamma='{self.gamma}', random_state={self.random_state})")


# Example usage and utility functions
def create_sample_data(n_inliers: int = 100, n_outliers: int = 20, 
                      n_features: int = 2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data for demonstration.
    
    Args:
        n_inliers: Number of inlier samples
        n_outliers: Number of outlier samples
        n_features: Number of features
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is the data and y are the labels
    """
    np.random.seed(random_state)
    
    # Generate inliers from a normal distribution
    X_inliers = np.random.randn(n_inliers, n_features) * 0.5
    
    # Generate outliers from a different distribution
    X_outliers = np.random.randn(n_outliers, n_features) * 2.0 + 3.0
    
    # Combine data
    X = np.vstack([X_inliers, X_outliers])
    y = np.hstack([np.ones(n_inliers), -np.ones(n_outliers)])
    
    return X, y


def demo_usage():
    """Demonstrate how to use the OneClassSVMClassifier class."""
    print("=== One-Class SVM Demo ===\n")
    
    # Create sample data
    X, y = create_sample_data(n_inliers=200, n_outliers=40, n_features=2)
    print(f"Data shape: {X.shape}")
    print(f"Number of inliers: {np.sum(y == 1)}")
    print(f"Number of outliers: {np.sum(y == -1)}\n")
    
    # Split data (in a real scenario, you'd only have inliers for training)
    # For demo purposes, we'll use all data for training
    X_train = X
    y_train = y
    
    # Create and train the model
    print("Training the model...")
    oc_svm = OneClassSVMClassifier(kernel='rbf', nu=0.1, gamma='scale', verbose=True)
    oc_svm.fit(X_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = oc_svm.predict(X_train)
    scores = oc_svm.score_samples(X_train)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    metrics = oc_svm.evaluate(X_train, y_train)
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Plot decision boundary
    print("\nPlotting decision boundary...")
    oc_svm.plot_decision_boundary(X_train, y_train)
    
    # Save and load model
    print("\nSaving and loading model...")
    oc_svm.save_model('demo_oc_svm.joblib')
    loaded_model = OneClassSVMClassifier.load_model('demo_oc_svm.joblib')
    
    # Verify loaded model works
    y_pred_loaded = loaded_model.predict(X_train)
    print(f"Predictions match: {np.array_equal(y_pred, y_pred_loaded)}")
    
    # Clean up
    if os.path.exists('demo_oc_svm.joblib'):
        os.remove('demo_oc_svm.joblib')
        print("Demo model file cleaned up")


if __name__ == "__main__":
    # Run the demo if the script is executed directly
    demo_usage()
