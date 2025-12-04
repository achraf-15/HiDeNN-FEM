import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import math

from src.test_LBFGS import FullBatchLBFGS 

from src.c_plots import plot_displacement_magnitude, plot_von_mises, plot_von_mises_tricontourfd

class TestOptimizer:
    def __init__(self, model: nn.Module, loss_fn):

        self.model = model
        self.loss_fn = loss_fn

        # Freeze coordinates
        self.model.freeze_coords()
        # Precompute geometry-dependent quantities
        self.model.precompute_Jaccobians()
        self.model.precompute_G_patch()
        #self.model.check_inverse()
        #self.model.gradient_check_inverse()

        # Storage for logging
        self.loss_history = []
        self.events = []  # list of dicts: {"epoch": int, "desc": str, "phase": str, "time": float}

        # Initialize global epoch counter
        self.global_epoch = 0

    def optimize(self, stages):

        # Track global time 
        global_start = time.time()

        for stage_idx, stage in enumerate(stages):
            phase = stage.get("optimizer")
            desc = stage.get("description", f"Stage {stage_idx+1}")
            epochs = stage.get("epochs", 50)
            # Choose optimizer
            phase_name = phase.lower()
            if phase_name in ["adam", "rmsprop", "sgd", 'adamw']:
                lr = stage.get("lr", 1e-2)
                
                # Base optimizer
                if phase_name == "adam":
                    opt = optim.Adam(self.model.parameters(), lr=lr)
                elif phase_name == "rmsprop":  # RMSProp
                    opt = optim.RMSprop(self.model.parameters(), lr=lr)
                elif phase_name == "sgd":
                    opt = optim.SGD(self.model.parameters(), lr=lr)
                else:
                    opt = optim.AdamW(self.model.parameters(), lr=lr)


            elif phase_name == "lbfgs":
                opt = FullBatchLBFGS (self.model.parameters(), lr=0.01, history_size=100, line_search='Wolfe', dtype=self.model.dtype, debug=True) 
            else:
                raise ValueError(f"Unsupported optimizer: {phase}")

            # Track time for this stage
            stage_start = time.time()

            # Epoch loop
            pbar = tqdm(range(epochs), desc=f"{phase.upper()} - {desc}")
            for epoch in pbar:
                self.global_epoch += 1

                def closure_fn():
                    opt.zero_grad()
                    loss = self.loss_fn(self.model)
                    return loss

                if phase_name.lower() != "lbfgs":
                    loss = closure_fn()
                    loss.backward()
                    opt.step()
                else:
                    loss = closure_fn()
                    loss.backward()
                    options = {
                        'closure': closure_fn,
                        'current_loss': loss,
                        'eps': 1e-10,    
                        'c1': 1e-4,
                        'c2': 0.9,        
                        'max_ls': 20,     
                        'ls_debug': False,
                    }
                    opt.step(options=options)

                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                self.loss_history.append({
                    "epoch": self.global_epoch,
                    "time": time.time() - global_start,
                    "loss": loss_val,
                    "phase": phase
                })

                if self.global_epoch % max(1, epochs // 50) == 0 or epoch == epochs-1:
                    pbar.set_postfix({"loss": f"{loss_val:.6e}"})

                # if self.global_epoch % max(1, epochs // 10) == 0 or epoch == epochs-1:
                #     plot_displacement_magnitude(self.model)
                #     plot_von_mises(self.model)
                #     plot_von_mises_tricontourfd(self.model)

            stage_time = time.time() - stage_start
            self.events.append({
                "epoch": self.global_epoch,
                "desc": desc,
                "phase": phase,
                "time": stage_time
            })

    def plot_loss(self):
        """
        Plot relative error history with phase separation lines from events
        """
        epochs = np.array([h["time"] for h in self.loss_history])
        losses = np.array([h["loss"] for h in self.loss_history])

        # Compute relative error (one value shorter)
        rel_error = np.abs((losses[1:] - losses[:-1]) / losses[1:])
        # X-axis for relative change is shifted by one
        rel_epochs = epochs[1:]

        plt.figure(figsize=(10, 5))
        plt.semilogy(rel_epochs, rel_error, label="Loss", color='blue')
        #plt.plot(epochs, losses, label="Loss", color='blue')

        # Add vertical lines for events
        t = 0
        for event in self.events:
            t += event["time"]
            plt.axvline(t, linestyle='--', color='red', alpha=0.5)
            plt.text(t, max(losses), event["desc"], rotation=90, verticalalignment='top', fontsize=8)

        plt.xlabel("Excution time")
        plt.ylabel("Relative Loss")
        plt.title("Relative Loss evolution over time")
        plt.grid(True)
        plt.tight_layout(pad=1.5)
        plt.show()

