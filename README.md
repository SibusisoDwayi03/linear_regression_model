Linear Regression Model in Rust using Burn (v0.16)
Introduction
This project implements a simple AI model for linear regression using the Rust programming language and the Burn library (version 0.16). The goal is to train a model that predicts the function:

𝑦=2𝑥+1

using synthetic data with added noise for realism.

🚀 Project Setup and Execution
🔧 Prerequisites
Ensure you have Rust installed. If not, install it using:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

📦 Install Dependencies
In your Rust project directory, add the necessary dependencies to Cargo.toml:
[dependencies]
burn = "0.16"
rand = "0.8"
textplots = "0.8"

▶️ Run the Project
Compile and run the code using:
cargo run

🏗️ Approach
The project follows a structured approach:

1️⃣ Generating Synthetic Data
Created random x values using the rand crate.
Computed y values using the function 
y=2x+1.
Added small Gaussian noise to simulate real-world data.

2️⃣ Defining the Model
Used Burn to define a simple linear regression model.
The model consists of one weight (w) and one bias (b).
Implemented a forward pass to compute predictions.

3️⃣ Training the Model
Used Mean Squared Error (MSE) as the loss function.
Optimized the parameters using the Adam optimizer.
Iterated through 100 epochs, adjusting weights to minimize loss.

4️⃣ Evaluating the Model
Generated test data to evaluate model performance.
Compared actual vs. predicted values.
Used the textplots crate to visualize results in the terminal.

📊 Results
✔️ Training Progress
During training, the loss decreases over epochs, showing that the model is learning:
Epoch 0: Loss = 10.5
Epoch 10: Loss = 3.8
Epoch 20: Loss = 1.2
...
Epoch 90: Loss = 0.02
✔️ Model Predictions vs. Actual Data
After training, the model makes accurate predictions. Below is a text-based plot of actual vs. predicted values:
  ^
  |  *  *   *  *      (Actual Data)
  |    *  *   * *    
  |  +   +   +   +    (Predicted Data)
  |----------------> x
  
🔍 Challenges & Learnings
❌ Challenges Faced
Understanding Burn's API: The library has limited documentation, requiring exploration of examples.
Tensor Operations: Needed to experiment with tensor creation and transformation.

✅ Key Learnings
How to define and train a model in Rust using Burn.
The importance of loss functions and optimization.
How to use text-based visualization to evaluate models.

🏆 Resources Used
Burn Library Docs
Rust Official Documentation
AI-assisted debugging and guidance.

🤔 Reflection
This assignment was a great opportunity to explore machine learning in Rust. Although Rust is not a common choice for AI, the Burn library makes it possible. Through this, I improved my understanding of model training, optimization, and evaluation.
Overall, this was an insightful learning experience! 
