import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import math

def linear_warmup(epoch, total_epochs, lr_init, lr_target):
    return lr_init + (lr_target - lr_init) * (epoch / total_epochs)

def exponential_warmup(epoch, total_epochs, lr_init, lr_target):
    ratio = (epoch / total_epochs)
    return lr_init * ((lr_target / lr_init) ** ratio)

def cosine_warmup(epoch, total_epochs, lr_init, lr_target):
    return lr_init + 0.5 * (lr_target - lr_init) * (1 - math.cos(math.pi * epoch / total_epochs))

def polynomial_warmup(epoch, total_epochs, lr_init, lr_target, power=2):
    ratio = (epoch / total_epochs) ** power
    return lr_init + (lr_target - lr_init) * ratio

def no_warmup(epoch, total_epochs, lr_init, lr_target):
    return lr_init

warmup_dict = {
    'linear': linear_warmup,
    'exponential': exponential_warmup,
    'cosine': cosine_warmup,
    'polynomial': polynomial_warmup,
    'None': no_warmup
}




class HybridOptimizer:
    def __init__(self, model: nn.Module, loss_fn, device=None, dtype=torch.float32):

        self.model = model
        self.loss_fn = loss_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Freeze coordinates
        self.model.freeze_coords()
        # Precompute geometry-dependent quantities
        self.model.precompute_Jaccobians()
        self.model.precompute_G_patch()

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
            if phase_name in ["adam", "rmsprop"]:
                lr_init = stage.get("lr_init", 1e-5)
                
                warmup = stage.get("warmup", 'None')
                lr_target = stage.get("lr_target", lr_init)
                warmup_epochs = stage.get("warmup_epochs", 0)
                assert epochs >= warmup_epochs

                decay = stage.get("decay", 'None')
                lr_decay = stage.get("lr_decay", lr_target)
                decay_epochs = epochs - warmup_epochs

                
                # Base optimizer
                if phase_name == "adam":
                    opt = optim.Adam(self.model.parameters(), lr=lr_init)
                else:  # RMSProp
                    opt = optim.RMSprop(self.model.parameters(), lr=lr_init)


            elif phase_name == "lbfgs":
                opt = optim.LBFGS(self.model.parameters(), line_search_fn ="strong_wolfe")
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
                    loss.backward()
                    return loss

                if phase.lower() != "lbfgs":

                    if epoch < warmup_epochs:
                        lr = warmup_dict[warmup](epoch, warmup_epochs, lr_init, lr_target)
                    else:
                        decay_epoch = epoch - warmup_epochs
                        decay_epochs = decay_epochs 
                        lr = warmup_dict[decay](decay_epoch, decay_epochs, lr_target, lr_decay)

                    for g in opt.param_groups:
                        g['lr'] = lr
                    loss = closure_fn()
                    opt.step()

                else:
                    # LBFGS requires a closure returning loss
                    loss = opt.step(closure_fn)

                loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                self.loss_history.append({
                    "epoch": self.global_epoch,
                    "time": time.time() - global_start,
                    "loss": loss_val,
                    "phase": phase
                })

                if self.global_epoch % max(1, epochs // 50) == 0 or epoch == epochs-1:
                    pbar.set_postfix({"loss": f"{loss_val:.6e}"})
                    #self.model.test_conditioning()

            stage_time = time.time() - stage_start
            self.events.append({
                "epoch": self.global_epoch,
                "desc": desc,
                "phase": phase,
                "time": stage_time
            })

    def plot_loss(self):
        """
        Plot loss history with phase separation lines from events
        """
        epochs = [h["time"] for h in self.loss_history]
        losses = [h["loss"] for h in self.loss_history]

        plt.figure(figsize=(10, 5))
        #plt.semilogy(epochs, losses, label="Loss", color='blue')
        plt.plot(epochs, losses, label="Loss", color='blue')

        # Add vertical lines for events
        t = 0
        for event in self.events:
            t += event["time"]
            plt.axvline(t, linestyle='--', color='red', alpha=0.5)
            plt.text(t, max(losses), event["desc"], rotation=90, verticalalignment='top', fontsize=8)

        plt.xlabel("Excution time")
        plt.ylabel("Loss")
        plt.title("Loss evolution over time")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
