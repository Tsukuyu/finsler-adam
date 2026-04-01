"""
Finsler-Adam: Minimal Example (10 lines)

Usage:
    pip install finsler-adam
    python minimal.py
"""

import torch
import torch.nn as nn
from finsler_adam import FinslerAdam

model = nn.Linear(10, 1)
optimizer = FinslerAdam(model.parameters(), lr=1e-3, gamma=0.5, anna_alpha=0.1)

for step in range(100):
    x = torch.randn(32, 10)
    loss = model(x).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

print("Done.")
