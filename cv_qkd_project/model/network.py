from __future__ import annotations

import torch
from torch import nn


class VAPredictor(nn.Module):
    """
    MLP that predicts log(V_A*) from standardized channel/hardware features.

    Input features (dim=4):
      [T, xi, eta1, eta2] after standardization using `data/processed/scaler.npy`.

    Output:
      log(V_A*) (natural log), so the model is trained with MSE in log-space.
    """

    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 4), float32.

        Returns
        -------
        torch.Tensor
            Shape (batch,), predicted log(V_A*).
        """
        y = self.net(x)
        return y.squeeze(-1)

    @torch.no_grad()
    def predict_VA(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience prediction in linear space.

        Returns exp(forward(x)) which corresponds to V_A* in SNU.
        """
        return torch.exp(self.forward(x))


if __name__ == "__main__":
    m = VAPredictor()
    x = torch.randn(8, 4)
    y = m(x)
    print("log(VA*) shape:", y.shape)
    print("VA* shape:", m.predict_VA(x).shape)

