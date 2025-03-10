# Single Layer 1D Linear Regression

This example demonstrates linear regression using a single Dense layer in Keras on synthetic 1D data, based on the equation:

$$
y = 3x + \epsilon
$$

where  
- $$x \in \mathbb{R}$$ is a single scalar, and the complete dataset is stored as a column vector $$X$$ with shape $$(200, 1)$$, i.e., there are 200 samples with one feature each.
- $$y \in \mathbb{R}$$ is the scalar output, and  
- $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$ represents Gaussian noise.

Files:  
- **data.py:** Generates synthetic training and testing data.  
- **model.py:** Defines and compiles the neural network.  
- **plot.py:** Visualizes training, testing, and prediction results.
- **main.py:** Orchestrates data generation, model training, prediction, and plotting.

To run:  
1. Install dependencies:  
   ```
   pip install numpy matplotlib tensorflow scikit-learn
   ```
2. Execute:  
   ```
   python main.py
   ```
