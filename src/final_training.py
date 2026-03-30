import torch
#upgraded parameters
N_EPOCHS = 40
emb_dim = 128
latent_dim = 256
num_layers = 2
lr = 0.0005
max_seq_len = 150
batch_size = 8
dropout = 0.1

best_val_loss = float('inf')
patience = 5
counter = 0
# Feature Toggles
USE_SCHEDULER = True
USE_GRAD_CLIP = True
USE_EARLY_STOPPING = True

# Early Stopping Params
patience = 5
min_delta = 0.001

# Gradient Clipping Param
max_grad_norm = 1.0


# Scheduler Factory
def get_scheduler(optimizer):
    if USE_SCHEDULER:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
    return None


# Gradient Clipping
def apply_gradient_clipping(model):
    if USE_GRAD_CLIP:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # continue training
        else:
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                print("Early stopping triggered")
                return True  # stop training
            return False