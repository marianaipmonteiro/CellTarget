import numpy as np
import torch
import torch.optim as optim
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal
from scipy.stats import qmc
import math
#from space_fillers import generate_sobol_points

class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def generate_sobol_points(lower_bound, upper_bound, num_points):
    dim = len(lower_bound)
    sampler = qmc.Sobol(d=dim, scramble=True)
    sample = sampler.random_base2(m=int(np.ceil(np.log2(num_points))))
    return qmc.scale(sample[:num_points], lower_bound, upper_bound)


def bayesian_optimization(
    f,
    lower_bound: torch.Tensor,          # shape [d]
    upper_bound: torch.Tensor,          # shape [d]
    num_initial_points: int,
    num_iterations: int,
) -> torch.Tensor:                      # returns shape [d]
    # Sobol points in [n, d]
    train_x: torch.Tensor = torch.tensor(
        generate_sobol_points(lower_bound, upper_bound, num_initial_points),
        dtype=torch.float32,
    )  # [n, d]

    train_y: torch.Tensor = torch.stack([f(train_x[i, :]).squeeze() for i in range(train_x.shape[0])])  # [n]

    likelihood = GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    fit_gpytorch_model(likelihood, model, train_x, train_y)

    for _ in range(num_iterations):
        candidate_x: torch.Tensor = select_next_point(model, lower_bound, upper_bound)   # [d]
        candidate_y: torch.Tensor = f(candidate_x).reshape(-1,)                          # [1]

        train_x = torch.cat([train_x, candidate_x.unsqueeze(0)], dim=0)                  # [n+1, d]
        train_y = torch.cat([train_y, candidate_y], dim=0)                               # [n+1]

        likelihood = GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood)
        fit_gpytorch_model(likelihood, model, train_x, train_y)

    best_index: int = int(torch.argmin(train_y).item())
    return train_x[best_index, :]



def original_bayesian_optimization(f, lower_bound, upper_bound, num_initial_points, num_iterations):
    train_x = torch.tensor(generate_sobol_points(lower_bound, upper_bound, num_initial_points)).float().T

    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
    print(f"Shape of train_x: {train_x.shape}")  # Print the shape of train_x for debugging

    train_x = train_x[:, 0]  # Take the first candidate for simplicity TODO: check

    train_y = f(train_x).squeeze()
    print(f"Shape of train_y: {train_y.shape}")  # Add this to check the shape

    likelihood = GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    print(f"Shape of train_x: {train_x.shape}, Shape of train_y: {train_y.shape}")

    fit_gpytorch_model(likelihood, model, train_x, train_y)

    print('initial min:', train_y.min().item())

    for _ in range(num_iterations):
        candidate_x = select_next_point(model, lower_bound, upper_bound)
        candidate_y = f(candidate_x).reshape(-1,)

        train_x = torch.cat([train_x, candidate_x])
        train_y = torch.cat([train_y, candidate_y])

        model.set_train_data(train_x, train_y, strict=False)

        # create a new model and fit it NOTE there is a better way to do this, by reinitializing the model
        likelihood = GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood)
        fit_gpytorch_model(likelihood, model, train_x, train_y)

    print('final min:', train_y.min().item())

    return model


def fit_gpytorch_model(
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        model: gpytorch.models.ExactGP,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
) -> None:
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    optimizer = optim.LBFGS(model.parameters(), lr=1e-4, line_search_fn="strong_wolfe")
    model.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(likelihood, model)

    tolerance: float = 1e-6
    previous_loss: float | None = None

    # ✅ FIX: Disable debug mode to bypass strict input equality checks (which fail on NaNs)
    with gpytorch.settings.debug(False):
        for k in range(1000):
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                # Use *model.train_inputs to be safe
                output = model(*model.train_inputs)
                loss = -mll(output, model.train_targets)
                loss.backward()
                return loss

            # LBFGS step
            try:
                optimizer.step(closure)
            except RuntimeError as e:
                # Catch numerical errors during optimization (common if NaNs exist)
                print(f"Optimizer step failed: {e}")
                break

            # Convergence check
            with torch.no_grad():
                output = model(*model.train_inputs)
                loss_val: float = -mll(output, model.train_targets).item()

            if previous_loss is not None and abs(loss_val - previous_loss) <= tolerance:
                # print(f"Convergence achieved after {k+1} iterations.")
                break
            previous_loss = loss_val

    likelihood.eval()
    model.eval()


def old_fit_gpytorch_model(
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    model: gpytorch.models.ExactGP,
    train_x: torch.Tensor,   # [n, d]
    train_y: torch.Tensor,   # [n]
) -> None:
    train_x_fit: torch.Tensor = train_x.detach().contiguous().clone()
    train_y_fit: torch.Tensor = train_y.detach().contiguous().clone()

    # ✅ set the SAME tensors you will use in forward passes
    model.set_train_data(inputs=train_x_fit, targets=train_y_fit, strict=False)

    optimizer = optim.LBFGS(model.parameters(), lr=1e-4, line_search_fn="strong_wolfe")
    model.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(likelihood, model)

    tolerance: float = 1e-6
    previous_loss: float | None = None

    for k in range(1000):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x_fit)                 # matches stored train_inputs
            loss = -mll(output, train_y_fit)
            loss.backward()
            return loss

        optimizer.step(closure)

        # ✅ evaluate consistently on the same tensors
        loss_val: float = (-mll(model(train_x_fit), train_y_fit)).item()

        if previous_loss is not None and abs(loss_val - previous_loss) <= tolerance:
            print(f"Convergence achieved. after {k+1} iterations.")
            break
        previous_loss = loss_val

    likelihood.eval()
    model.eval()


def select_next_point(model, lower_bound, upper_bound, acq:str='ucb'):
    # Implement your acquisition function here
    # For example, you can use Expected Improvement (EI)
    # EI = (mu - f_best) * Phi(Z) + sigma * phi(Z)
    # where mu is the predicted mean, f_best is the best observed value so far,
    # sigma is the predicted standard deviation, Z = (mu - f_best) / sigma,
    # Phi is the cumulative distribution function of the standard normal distribution,
    # and phi is the probability density function of the standard normal distribution.
    # Return the point with the maximum acquisition value.

    if acq == 'ucb':
        return ucb(model, lower_bound, upper_bound)
    else:
        raise ValueError(f"Invalid acquisition function: {acq}")


def ucb(model, lower_bound, upper_bound, beta=2.0, num_restarts=5):
    # Ensure the bounds are tensors of shape [26]
    bounds = torch.stack([lower_bound, upper_bound], dim=1)  # Shape [26, 2]

    # Define the acquisition function (Upper Confidence Bound)
    def acquisition(x):
        model.eval()
        with gpytorch.settings.fast_pred_var():
            # Reshape x to match the expected input size (should be [1, 26])
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Ensure x has shape [1, 26]

            # Predict mean and variance for x
            pred = model(x)
            mean = pred.mean
            variance = pred.variance

            # UCB formula: mean + beta * variance
            # ORIGINAL
            # return - (mean + 1.96 * torch.sqrt(variance))  # Maximize UCB (negative for minimizing)
            return (mean - beta * torch.sqrt(variance)).sum()  # <-- LCB

    # Initial candidate point
    x0 = (lower_bound + upper_bound) / 2
    x0 = x0.clone().detach().requires_grad_(True)

    # Use LBFGS optimizer to optimize the acquisition function
    optimizer = torch.optim.LBFGS([x0], max_iter=20)

    def closure():
        optimizer.zero_grad()
        loss = acquisition(x0)
        loss.backward()
        return loss

    # Perform optimization step
    optimizer.step(closure)

    # Ensure x0 stays within bounds after optimization by clipping
    x0 = torch.clamp(x0, lower_bound, upper_bound)

    # Check the shape after optimization (should be [26])
    print(f"Shape of candidate_x after optimization: {x0.shape}")
    return x0.detach()



def original_ucb(model, lower_bound, upper_bound):
    # Implement the Upper Confidence Bound (UCB) acquisition function


    def ucb_acquisition(x):
        # Implement the UCB acquisition function
        # UCB = mu + beta * sigma
        # where mu is the predicted mean, sigma is the predicted standard deviation,
        # and beta is a hyperparameter that controls the trade-off between exploration and exploitation. 

        # this is defined to optimistically select minimas
        pred  = model(x)
        beta = 2
        return pred.mean - beta * pred.stddev
    
    

    # Create the LBFGS optimizer
    n_multi_start = 10
    n_grad_steps = 50
    points = generate_sobol_points(lower_bound, upper_bound, n_multi_start).T
    minima = []
    sol_x  = []
    tolerance = torch.tensor([1e-8])  # Define your tolerance threshold
    previous_acquisition_value = None

    for i in range(n_multi_start):
        next_point = torch.tensor(points[i].reshape(1,-1)).float().requires_grad_(True)
        optimizer = optim.LBFGS([next_point], line_search_fn='strong_wolfe', lr=0.01)

        # Optimize the UCB acquisition function
        for j in range(n_grad_steps):
            # Define the closure function for LBFGS optimizer
            def closure():
                optimizer.zero_grad()
                acquisition_value = ucb_acquisition(next_point)
                acquisition_value.backward()
                return acquisition_value
            
            optimizer.step(closure)
            with torch.no_grad(): next_point[:] = next_point.clamp(lower_bound, upper_bound)

            # Get the current acquisition value
            current_acquisition_value = ucb_acquisition(next_point).float().detach().clone().requires_grad_(False)

            # Check for convergence
            if previous_acquisition_value is not None:
                change_in_value = torch.absolute(current_acquisition_value - previous_acquisition_value)
                if change_in_value <= tolerance:
                    print(f"Convergence achieved after {j+1} iterations.")
                    break  # Exit the loop if convergence is achieved

            # Update the previous_acquisition_value for the next iteration
            previous_acquisition_value = current_acquisition_value

        minima.append(current_acquisition_value)
        sol_x.append(next_point.clone().detach())

    # Select the point with the minimum acquisition value
    minima = torch.tensor(minima)
    min_idx = torch.argmin(minima)
    next_point = sol_x[min_idx]
    return next_point


if __name__ == '__main__':
    # Example usage
    def f(x):
        return torch.sin(x) 
    
    def rastrigin(x, A=1):
        return A * x.size(-1) + (x**2 - A * torch.cos(2 * math.pi * x)).sum(dim=-1)

    lower_bound = -2.45 * math.pi
    upper_bound = 8 * math.pi
    num_initial_points = 2
    num_iterations = 30

    model = bayesian_optimization(f, lower_bound, upper_bound, num_initial_points, num_iterations)


    