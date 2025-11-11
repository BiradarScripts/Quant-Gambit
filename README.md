# Market State Forecasting Model

This project's goal is to develop a machine learning model that predicts the next market state vector based on a sequence of prior states. The model will be built to specification for a sequence modeling challenge.

## 🎯 Project Goal

The main task is to build a stateful model that:
1.  Receives market state data, one step at a time.
2.  Maintains an internal "memory" or "state" (e.g., the hidden state of an RNN).
3.  Resets this internal state whenever a new, independent sequence begins.
4.  Outputs a prediction vector (of N features) only when requested.

## 💾 Dataset

The training data for this competition can be found on Kaggle:

* **Kaggle Dataset:** [https://www.kaggle.com/datasets/biradar1913/wunder](https://www.kaggle.com/datasets/biradar1913/wunder)

## ✅ Key Tasks: What Must Be Done

Your entire solution must be implemented within a single Python class named `PredictionModel`, which will be saved in a file called `solution.py`.

This class has two primary responsibilities:

### 1. Initialization (`__init__`)
* This is where you'll load your trained model, initialize any internal state variables, and set up your prediction pipeline.
* For example, you might load model weights, define a `self.current_seq_id = None`, and initialize `self.model_state = None`.

### 2. Prediction (`predict`)
This method is the core of the project. It will be called repeatedly, once for each row (or `DataPoint`) in the dataset.

Your logic *must* handle the following:

* **State Management:** Check if the `data_point.seq_ix` is new.
    * If it is, you **must** reset your model's internal state (e.g., clear the RNN/LSTM hidden state). This is critical because sequences are independent.
* **Prediction Trigger:** Check the `data_point.need_prediction` flag.
    * If `False`: You should still update your model's internal state using the `data_point.state`, but you must return `None`.
    * If `True`: You must:
        1.  Generate the prediction for the *next* step (based on the history you've tracked).
        2.  Return the prediction as a `numpy.ndarray` with shape `(N,)`.
        3.  Update your model's internal state with the *current* `data_point.state` to be ready for the next call.

## 📝 Required File: `solution.py`

Your main deliverable is this file. It must contain the `PredictionModel` class with the exact structure below. All your logic, model loading, and state management must be built into this class.

```python
import numpy as np
from utils import DataPoint # Assuming utils.py is provided

class PredictionModel:
    
    def __init__(self):
        """
        Initialize your model here.
        - Load model weights (e.g., from a file included in your zip).
        - Initialize internal state trackers.
        """
        # Example:
        # self.model = self.load_my_model() 
        # self.current_seq_id = -1
        # self.model_state = None # e.g., for RNN hidden state
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next market state.
        
        Args:
            data_point: An object with attributes:
                - seq_ix (int): The ID for the current sequence.
                - step_in_seq (int): The step number within the sequence.
                CSS
                - need_prediction (bool): True if a prediction is required.
                - state (np.ndarray): The current market state vector (N features).
        
        Returns:
            - np.ndarray of shape (N,) if need_prediction is True.
            - None if need_prediction is False.
        """
        
        # --- 1. State Management: Reset if new sequence ---
        if data_point.seq_ix != self.current_seq_id:
            # self.current_seq_id = data_point.seq_ix
            # self.reset_model_state() # Call your state reset logic
            pass
        
        # --- 2. Check if prediction is needed ---
        prediction_output = None
        if data_point.need_prediction:
            # --- 3. Generate Prediction ---
            # Use self.model_state and data_point.state to predict NEXT state
            # prediction_output = self.model.predict(self.model_state, ...)
            
            # Placeholder: Replace with your model's actual output
            prediction_output = np.zeros(data_point.state.shape)
        
        # --- 4. Update Internal State ---
        # Feed the *current* state to your model to update its
        # internal state for the *next* call.
        # self.model_state = self.model.update(self.model_state, data_point.state)
        pass

        # --- 5. Return Prediction (or None) ---
        return prediction_output
```
## 📊 Evaluation

The model's performance will be judged using the **R² (coefficient of determination)** score. A higher R² score is better.

The score is calculated for each of the N features *i* using the formula:

$R^2_i = 1 - \frac{\sum(y_{\text{true}} - y_{\text{pred}})^2}{\sum(y_{\text{true}} - y_{\text{mean}})^2}$

The final competition score is the **average of the R² scores** across all N features.

You can use the `utils.py` file (if provided) to calculate this score locally and validate your model's performance.

## 🚀 Recommended Workflow

1.  **Explore:** Use a Jupyter notebook to load and analyze `datasets/train.parquet`.
2.  **Validate:** Create a local validation set. Since sequences are independent, you can split them by `seq_ix` (e.g., 80% for training, 20% for validation).
3.  **Develop:** Train a sequence model (like an LSTM, GRU, or Transformer) on your training set.
4.  **Implement:** Transfer your trained model's logic into the `PredictionModel` class in `solution.py`.
5.  **Package:** Create a `.zip` file containing `solution.py` and any necessary supporting files (like `.h5` or `.pth` model weights). Ensure `solution.py` is at the root of the zip.

