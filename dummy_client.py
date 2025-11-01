import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import sys
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Define the Model
# We use a simple Logistic Regression model.
# The server will never know what *kind* of model this is.
model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # Keep training local and fast
    warm_start=True, # Allow retraining
)

def load_data():
    """
    Generates dummy data for this client.
    In a real app, this would load the user's actual local data.
    """
    print("Loading dummy data for Client (Mother B)...")
    # Generate 100 samples with 2 features (e.g., 'heart_rate', 'emotion_score')
    X_train = np.random.rand(100, 2) 
    # Generate 100 binary labels (e.g., 0 = Not Stressed, 1 = Stressed)
    y_train = np.random.randint(0, 2, 100)
    
    # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    
    return (X_train, y_train), (None, None) # No test set for this simple client

# 2. Define the Flower Client (NumPyClient)
class DummyClient(fl.client.NumPyClient):
    """
    This is the client-side implementation for our dummy client.
    """
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        # This is a trick to set the initial model parameters
        self.model.fit(X_train, y_train) 

    def get_parameters(self, config):
        """Gets the current local model parameters."""
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        """Sets the local model parameters from the server."""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        """
        Trains the local model on the local data.
        
        Args:
            parameters: The model parameters sent by the server.
            config: Configuration data (not used here).
            
        Returns:
            The updated model parameters, number of examples trained on,
            and an empty metrics dictionary.
        """
        print("\n[Client] Training local model (fit)...")
        # 1. Set the model parameters from the server
        self.set_parameters(parameters)
        
        # 2. Train the model on local data
        self.model.fit(self.X_train, self.y_train)
        
        print("[Client] Training complete.")
        
        # 3. Return the updated parameters to the server
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluates the local model.
        
        Args:
            parameters: The model parameters sent by the server.
            config: Configuration data (not used here).
            
        Returns:
            The loss, number of examples, and an accuracy metric.
        """
        print("[Client] Evaluating local model (evaluate)...")
        # 1. Set the model parameters
        self.set_parameters(parameters)
        
        # 2. Evaluate the model
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        accuracy = self.model.score(self.X_train, self.y_train)
        
        print(f"[Client] Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # 3. Return the results
        return loss, len(self.X_train), {"accuracy": accuracy}

def main():
    """
    Loads data, instantiates the client, and starts the connection.
    """
    print("Starting Dummy Client (Mother B)...")
    
    # 1. Load data
    (X_train, y_train), _ = load_data()
    
    # 2. Instantiate the client
    client = DummyClient(model, X_train, y_train)

    # 3. Start the client
    # It will try to connect to the server running on 127.0.0.1:8080
    try:
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=client
        )
    except Exception as e:
        print(f"\n[ERROR] Could not connect to server: {e}")
        print("Please make sure the 'server.py' is running in another terminal.")
        sys.exit(1)

if __name__ == "__main__":
    main()
