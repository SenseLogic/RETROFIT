![](https://github.com/senselogic/RETROFIT/blob/master/LOGO/retrofit.png)

# Retrofit

Old-school stock price prediction using Levenberg-Marquardt optimized linear regression with rolling windows.

## Overview

This script performs one-step-ahead stock price prediction by combining:
- Multi-lag price-to-price mapping (predicting p_{t+1} from [p_t, p_{t-1}, p_{t-2}])
- A multi-feature linear model (y = a₀·x₀ + a₁·x₁ + a₂·x₂ + b)
- Levenberg–Marquardt (LM) optimization, implemented via `scipy.optimize.least_squares(method='lm')`
- Iterative, rolling-window style forecasting of a configurable number of future points
- Residual noise injection to preserve historical volatility patterns

![](https://github.com/senselogic/RETROFIT/blob/master/TEST/FUTURE/future_apple_data.svg)
![](https://github.com/senselogic/RETROFIT/blob/master/TEST/FUTURE/future_google_data.svg)
![](https://github.com/senselogic/RETROFIT/blob/master/TEST/FUTURE/future_microsoft_data.svg)
![](https://github.com/senselogic/RETROFIT/blob/master/TEST/FUTURE/future_nike_data.svg)
![](https://github.com/senselogic/RETROFIT/blob/master/TEST/FUTURE/future_nvidia_data.svg)

## Input Format

A CSV file with at least two columns:
- A date column (parsable as a timestamp)
- A price column (the closing price of a single stock for each trading day)

The column names are specified via command-line arguments.

## Time Series Representation

Let p_t denote the closing price at date t, where t = 1, 2, ..., T.

## Feature Engineering

A multi-lag approach is used to better capture trends and volatility:
- For each time step t, three lagged prices are used as input features: [p_t, p_{t-1}, p_{t-2}]
- The target is the next price p_{t+1}
- This creates training pairs: ([p_t, p_{t-1}, p_{t-2}], p_{t+1}) for t = 3, 4, ..., T-1

The feature matrix X has shape (T-3, 3), where each row contains three consecutive price values.
The target vector y has shape (T-3, 1), containing the corresponding next prices.

## Normalization

Before feeding X and the targets y into the model, they are normalized:
- For features: X_norm = (X - mean_X) / std_X (with small epsilon to avoid division by zero)
- For targets: y_norm = (y - mean_y) / std_y (scalar mean/std for the current training window)

Normalization helps:
- Keep gradients and residuals numerically well-behaved
- Make the LM linear system better conditioned
- Improve convergence stability

## Model: Multi-Feature Linear Regression

A multi-feature linear model is used:
- Input: x ∈ R³ (three normalized lagged price values: [p_t, p_{t-1}, p_{t-2}])
- Output: ŷ = a₀·x₀ + a₁·x₁ + a₂·x₂ + b, where a₀, a₁, a₂, and b are parameters to be optimized

For a batch of inputs X ∈ R^{N × 3} (N samples), the computation is:
- Ŷ = X @ [a₀, a₁, a₂]ᵀ + b (broadcasted over all samples)

Parameters are stored as a vector θ = [a₀, a₁, a₂, b], which LM optimizes to minimize the sum of squared residuals.

The model is initialized with a₀=1, a₁=0, a₂=0, b=0 (identity mapping for the first feature) for determinism, ensuring reproducible results.

## Loss Function and Nonlinear Least Squares

The objective function is defined as the sum of squared errors (SSE) between predicted and actual (normalized) prices in the training window.

For N training samples:
- Input matrix: X_norm ∈ R^{N × 3}
- Target vector: y_norm ∈ R^{N × 1}
- Model output: ŷ_norm(X_norm; θ) = X_norm @ [a₀, a₁, a₂]ᵀ + b ∈ R^{N × 1}

Residuals for LM:
- r_i(θ) = ŷ_norm_i(θ) - y_norm_i, for i = 1..N
- The SSE objective is: E(θ) = (1/2) ∑_{i=1}^N [r_i(θ)]²

Gradients and Jacobians are not computed explicitly; instead, the residual vector r(θ) is provided to `scipy.optimize.least_squares` with `method='lm'`. SciPy numerically approximates the Jacobian of r with respect to θ.

## Levenberg–Marquardt (LM) Optimization

LM is a standard algorithm to minimize SSE in nonlinear least squares problems. It interpolates between:
- Gradient descent (robust far from the minimum)
- Gauss–Newton (fast near the minimum if the model is close to locally linear in parameters)

At each iteration k, LM solves a damped normal equation:
- (J^T J + λ I) Δθ = -J^T r

where:
- θ is the parameter vector
- r is the residual vector r(θ)
- J is the Jacobian matrix of r with respect to θ, J_{ij} = ∂r_i / ∂θ_j
- λ is a damping parameter (adaptively tuned in classic LM; SciPy handles this internally)

**Interpretation:**
- For λ large, the (J^T J + λ I) term ≈ λ I, and the step direction approximates steepest descent (gradient descent) with small step size
- For λ small, the method approximates Gauss–Newton, which uses a second-order approximation and can converge rapidly when close to the solution

**SciPy's `least_squares(method='lm')`:**
- Expects a residual function f(θ) ∈ R^m (m data points)
- Minimizes (1/2) ||f(θ)||²
- Computes/approximates J numerically and automatically adjusts damping

## Rolling / Iterative One-Step-Ahead Prediction

A configurable number of future predicted prices extending the existing series is generated.

Given T available observations (p_1, ..., p_T), the algorithm:
- For each forecast step k = 1..N (where N is the number of predictions requested):
  - Extract a training subseries of length up to `lookback` (default 252 trading days), ending at the current end of available data
  - Build feature-target pairs: X contains [p_t, p_{t-1}, p_{t-2}], y contains next prices p_{t+1}
  - Normalize X and y
  - Initialize a fresh linear model (a₀=1, a₁=0, a₂=0, b=0 for determinism)
  - Train via LM (`least_squares` with `method='lm'`)
  - Compute residual standard deviation from training data
  - Use the last available prices [p_T, p_{T-1}, p_{T-2}] as features to predict the next price
  - Add residual noise (sampled from N(0, σ_residual)) to preserve historical volatility
  - Unnormalize the predicted output to the original price scale
  - Append the prediction to the series for the next iteration

This produces N successive predictions, which are then appended as new rows with synthetic future dates (incrementing calendar days from the last observed date).

## Installation

```bash
pip install numpy pandas scipy matplotlib
```

## Command line

```bash
python retrofit.py <past_data_csv> <date_column_name> <price_column_name> <prediction_count> <OUTPUT_FOLDER_PATH/>
```

### Example

```bash
python retrofit.py microsoft_data.csv DATE CLOSE 90 FUTURE/
```

## Version

0.1

## Author

Eric Pelzer (ecstatic.coder@gmail.com).

## License

This project is licensed under the GNU General Public License version 3.

See the [LICENSE.md](LICENSE.md) file for details.
