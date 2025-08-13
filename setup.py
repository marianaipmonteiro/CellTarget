import cobra
import pandas as pd
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from typing import List
from cobra.io import load_json_model


class FullObservabilityCase:
    def __init__(self, num_fluxes: int, v_min: list, v_max: list, S: np.ndarray):
        self.num_fluxes = num_fluxes
        self.v_min = torch.tensor(v_min, dtype=torch.float32)
        self.v_max = torch.tensor(v_max, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)


class PartialObservabilityCase:
    def __init__(
        self,
        num_fluxes: int,
        v_min: list,
        v_max: list,
        S: np.ndarray,
        measured_flux_indices: list,
        mode: str = "sum_squares"
    ):
        self.num_fluxes = num_fluxes
        self.v_min = torch.tensor(v_min, dtype=torch.float32)
        self.v_max = torch.tensor(v_max, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)
        self.measured_flux_indices = measured_flux_indices
        self.mode = mode


class PartialObservabilityCase_original:
    def __init__(self, model_path: str, experimental_fluxes_path: str):
        self.model = load_json_model(model_path)
        self.experimental_fluxes_df = pd.read_csv(experimental_fluxes_path, index_col=0)
        self.experimental_fluxes_tensor = self.get_experimental_fluxes_tensor()
        self.S, self.v_min, self.v_max = self.initialize_stoichiometric_matrix_and_bounds()

    def get_experimental_fluxes_tensor(self):
        experimental_fluxes = self.experimental_fluxes_df.iloc[2]
        flux_array = torch.zeros(len(self.model.reactions))
        for i, reaction in enumerate(self.model.reactions):
            if reaction.id in experimental_fluxes.index:
                flux_array[i] = experimental_fluxes[reaction.id]
        return flux_array.clone().detach().requires_grad_(True)

    def initialize_stoichiometric_matrix_and_bounds(self):
        S = np.zeros((len(self.model.metabolites), len(self.model.reactions)))
        for i, metabolite in enumerate(self.model.metabolites):
            for j, reaction in enumerate(self.model.reactions):
                if metabolite in reaction.metabolites:
                    S[i, j] = reaction.metabolites[metabolite]
        v_min = torch.tensor([reaction.lower_bound for reaction in self.model.reactions], requires_grad=False)
        v_max = torch.tensor([reaction.upper_bound for reaction in self.model.reactions], requires_grad=False)
        return torch.tensor(S, dtype=torch.float32), v_min, v_max

    def get_reaction_indices(self, measured_flux_indices):
        valid_indices = []
        for flux_id in measured_flux_indices:
            try:
                valid_indices.append(self.model.reactions.index(self.model.reactions.get_by_id(flux_id)))
            except KeyError:
                print(f"Reaction ID {flux_id} not found in the model.")
        if not valid_indices:
            raise ValueError("No valid reaction indices found.")
        return valid_indices



class Optimizer:
    def __init__(self, fba_layer: CvxpyLayer, experimental_fluxes_tensor: torch.Tensor, v_min_torch: torch.Tensor, mask: torch.Tensor, measured_flux_indices: List[int], optimizer_type: str = 'LBFGS'):
        self.fba_layer = fba_layer
        self.experimental_fluxes_tensor = experimental_fluxes_tensor
        self.v_min_torch = v_min_torch
        self.mask = mask
        self.measured_flux_indices = measured_flux_indices
        self.C_torch = torch.ones((1, len(measured_flux_indices)), requires_grad=True)

        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam([self.C_torch], lr=0.00001)
        elif optimizer_type == 'LBFGS':
            self.optimizer = torch.optim.LBFGS([self.C_torch], lr=0.001)
        else:
            raise ValueError("Invalid optimizer type. Choose 'Adam' or 'LBFGS'.")

    def optimize(self, num_epochs: int = 50):
        loss_history = []
        coefficients_history = []

        for epoch in range(num_epochs):
            def closure():
                self.optimizer.zero_grad()
                predicted_fluxes, = self.fba_layer(self.C_torch)
                predicted_fluxes += self.v_min_torch
                exp_flux = predicted_fluxes.clone().detach().squeeze()

                for i in self.measured_flux_indices:
                    exp_flux[i] = self.experimental_fluxes_tensor[i]
                loss = torch.nn.functional.mse_loss(predicted_fluxes.squeeze(), exp_flux, reduction='none')
                loss = torch.sum(loss * self.mask)
                loss.backward()
                return loss

            self.optimizer.step(closure)
            loss_history.append(closure().item())
            coefficients_history.append(self.C_torch.clone().detach().cpu().numpy())
            print(f'Epoch {epoch}, Loss: {closure().item()}')

        return loss_history, coefficients_history