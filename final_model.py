import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import os
import time
from tqdm import tqdm
import math # For get_lr_scheduler
from graph import create_and_save_graph

FILENAME = 'DATAfile_case1_damagePattern1'

class WindowedAccelerationDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data, window_size=48, history_size=5, stride=16):
        """Dataset for windowed acceleration data with graph structure."""
        self.graph_data = graph_data
        self.window_size = window_size
        self.history_size = history_size
        self.stride = stride
        self.acc_data = graph_data.acceleration_data
        self.num_nodes = graph_data.num_nodes
        self.node_features = graph_data.x
        self.edge_index = graph_data.edge_index
        self.edge_attr = graph_data.edge_attr

        total_required_size = (history_size + 1) * window_size
        self.num_windows = (self.acc_data.shape[1] - total_required_size) // stride + 1

        print("Applying noise reduction...")
        kernel_size = 5
        padding = kernel_size // 2
        # Simple moving average filter
        for node_idx in range(self.num_nodes):
            smoothed = F.avg_pool1d(
                self.acc_data[node_idx].unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            ).squeeze()
            self.acc_data[node_idx] = smoothed

        self.acc_mean = torch.mean(self.acc_data)
        self.acc_std = torch.std(self.acc_data)
        print(f"Acceleration statistics - Mean: {self.acc_mean:.6f}, Std: {self.acc_std:.6f}")
        self.acc_data = (self.acc_data - self.acc_mean) / (self.acc_std + 1e-8) # Added epsilon for stability
        print(f"Normalized acceleration range: [{self.acc_data.min():.6f}, {self.acc_data.max():.6f}]")

        # Non-linear squashing to range (-2, 2) using tanh
        print('Applying tanh normalisation')
        scaling_factor = 3.0 
        self.acc_data = 2.0 * torch.tanh(self.acc_data / scaling_factor)
        self.acc_mean = torch.mean(self.acc_data)
        self.acc_std = torch.std(self.acc_data)
        print(f"Final Acceleration range: [{self.acc_data.min():.6f}, {self.acc_data.max():.6f}]")
        print(f"Final Acceleration statistics - Mean: {self.acc_mean:.6f}, Std: {self.acc_std:.6f}")

        print(f"Created dataset with {self.num_windows} windows, each with {window_size} timesteps")
        print(f"Using {history_size} historical windows with stride {stride}")

    def __len__(self):
        return self.num_windows * self.num_nodes # This dataset structure implies each node can be a target

    def __getitem__(self, idx):
        # node_idx here determines which node's history/target is being sampled as a primary focus for THIS data item
        # However, EnhancedDGAR model has fixed measured/unmeasured nodes.
        # The mask will be fixed, but history/target will relate to this idx.
        target_node_for_this_sample = idx % self.num_nodes
        window_idx = idx // self.num_nodes
        
        start_pos = window_idx * self.stride
        target_start = start_pos + self.history_size * self.window_size
        target_window_all_nodes = self.acc_data[:, target_start:target_start + self.window_size]

        history_windows_all_nodes = []
        for i in range(self.history_size):
            history_start = start_pos + i * self.window_size
            history_window = self.acc_data[:, history_start:history_start + self.window_size]
            history_windows_all_nodes.append(history_window)
        history_tensor_all_nodes = torch.stack(history_windows_all_nodes, dim=1) # Shape [num_total_nodes, history_size, window_size]

        # Fixed mask based on problem statement
        fixed_measured_nodes = [0, 3, 5, 6, 8, 11, 13, 14]
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[fixed_measured_nodes] = True
        
        data = Data(
            x=self.node_features.clone(), # Clone to avoid in-place modification issues if any
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone() if self.edge_attr is not None else None,
            history=history_tensor_all_nodes,   # History for all 16 nodes
            target=target_window_all_nodes,     # Target for all 16 nodes
            mask=mask,                          # Fixed mask
            target_node_idx=torch.tensor(target_node_for_this_sample, dtype=torch.long) # Original target for this sample
        )
        return data

class EnhancedDGAR(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, window_size=32,
                 history_size=5, hidden_dim=128, dropout=0.3):
        super(EnhancedDGAR, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.measured_nodes = [0, 3, 5, 6, 8, 11, 13, 14]
        self.unmeasured_nodes = [1, 2, 4, 7, 9, 10, 12, 15]
        
        self.node_embedding = torch.nn.Linear(node_feature_dim, hidden_dim)
        
        # Enhanced GNN with multiple layers
        self.gnn = GATConv(hidden_dim, hidden_dim, heads=4, concat=True,
                           edge_dim=edge_feature_dim, dropout=dropout, add_self_loops=False)
        gnn_out_dim = hidden_dim * 4  # Output dimension from GAT with concat=True
        
        # Improved projection layers
        self.gnn_proj = torch.nn.Sequential(
            torch.nn.Linear(gnn_out_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        
        # True temporal encoder for history
        self.history_encoder = torch.nn.GRU(
            input_size=window_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,  # Bidirectional to capture future dependencies
            num_layers=2,
        )
        # Project history encoder output to match dimensions (bidirectional = 2*hidden_dim)
        self.history_proj = torch.nn.Linear(hidden_dim*2, hidden_dim)

        # Cross attention between node embeddings and history embeddings
        self.cross_attn_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_v = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # Unified spatiotemporal processor for each unmeasured node
        self.st_processors = torch.nn.ModuleDict()
        for node_idx in self.unmeasured_nodes:
            # TCN (Temporal Convolutional Network) for better temporal modeling
            self.st_processors[f"tcn_{node_idx}"] = torch.nn.Sequential(
                torch.nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(hidden_dim*2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU()
            )
            
            self.st_processors[f"out_{node_idx}"] = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, window_size)
            )
        
        # Increase dropout and add spectral normalization
        self.dropout = 0.5  # Increase from 0.3
        self.spectral_norm = True
        self.feature_dropout = 0.2  # Add feature-level dropout

        # Apply spectral normalization to linear layers
        if self.spectral_norm:
            for node_idx in self.unmeasured_nodes:
                self.st_processors[f"out_{node_idx}"][0] = torch.nn.utils.spectral_norm(
                    self.st_processors[f"out_{node_idx}"][0]
                )
                self.st_processors[f"out_{node_idx}"][3] = torch.nn.utils.spectral_norm(
                    self.st_processors[f"out_{node_idx}"][3]
                )

    def forward(self, data):
        device = data.x.device
        
        # Basic node embeddings
        all_node_initial_embeds = F.relu(self.node_embedding(data.x))
        
        # Graph message passing
        current_edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        gnn_output_all_nodes = self.gnn(all_node_initial_embeds, data.edge_index, edge_attr=current_edge_attr)
        gnn_output_all_nodes = F.dropout(gnn_output_all_nodes, p=self.dropout, training=self.training)
        gnn_output_all_nodes = self.gnn_proj(gnn_output_all_nodes)  # Project to hidden_dim
        
        # Process batch information
        batch_assignment_vector = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        num_graphs_in_batch = batch_assignment_vector.max().item() + 1
        num_nodes_per_graph = data.x.size(0) // num_graphs_in_batch
        
        predictions_all = torch.zeros(data.x.size(0), self.window_size, device=device)
        
        for b in range(num_graphs_in_batch):
            node_indices_in_batch = torch.where(batch_assignment_vector == b)[0]
            
            # Get node embeddings and data for current graph
            node_embeds_from_gnn = gnn_output_all_nodes[node_indices_in_batch]
            history_for_graph = data.history[node_indices_in_batch]
            target_for_graph = data.target[node_indices_in_batch]
            
            # Process measured nodes with bidirectional history encoder
            measured_history_embeds = []
            for i, m_idx in enumerate(self.measured_nodes):
                history = history_for_graph[m_idx]
                output, _ = self.history_encoder(history.unsqueeze(0))
                # Take final state and project from bidirectional dimension
                history_embed = self.history_proj(output[:, -1])
                measured_history_embeds.append(history_embed.squeeze(0))
            
            measured_history_embeds = torch.stack(measured_history_embeds)
            
            # Initialize predictions with known values for measured nodes
            current_graph_predictions = torch.zeros(num_nodes_per_graph, self.window_size, device=device)
            current_graph_predictions[self.measured_nodes] = target_for_graph[self.measured_nodes]
            
            # Process each unmeasured node with improved spatiotemporal architecture
            for i, u_idx in enumerate(self.unmeasured_nodes):
                # Get node embedding from GNN
                u_embed = node_embeds_from_gnn[u_idx]
                
                # Cross-attention between unmeasured node and all measured nodes
                q = self.cross_attn_q(u_embed).unsqueeze(0)  # [1, hidden_dim]
                k = self.cross_attn_k(measured_history_embeds)  # [num_measured, hidden_dim]
                v = self.cross_attn_v(measured_history_embeds)  # [num_measured, hidden_dim]
                
                # Compute attention scores and weights
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float, device=device))
                attn_weights = F.softmax(attn_scores, dim=-1)  # [1, num_measured]
                
                # Compute context vector using attention
                context = torch.matmul(attn_weights, v)  # [1, hidden_dim]
                
                # Combine with unmeasured node embedding
                combined_feature = u_embed + context.squeeze(0)  # Residual connection
                
                # Process through temporal convolutional network (transposed for Conv1d)
                # TCN expects [batch, channels, sequence] - we use hidden_dim as sequence for the key timesteps
                tcn_input = combined_feature.unsqueeze(0).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                tcn_output = self.st_processors[f"tcn_{u_idx}"](tcn_input)
                
                # Average pool across the temporal dimension and pass through final projection
                pooled_output = tcn_output.mean(dim=2)  # [1, hidden_dim]
                prediction = self.st_processors[f"out_{u_idx}"](pooled_output.squeeze(0))
                
                current_graph_predictions[u_idx] = prediction
            
            predictions_all[node_indices_in_batch] = current_graph_predictions
            
        return predictions_all

def create_stratified_split(dataset, val_ratio=0.15, test_ratio=0.15):
    num_nodes = dataset.num_nodes # This is 16
    num_windows_per_node_type = dataset.num_windows # Total windows if we consider one node_type

    # Total samples in dataset is num_windows * num_nodes
    # The split should be on the windows, then duplicated for each node type concept in dataset
    
    indices = np.arange(len(dataset)) # Indices from 0 to (num_windows * num_nodes - 1)
    
    # We need to split based on window_idx to keep temporal consistency
    # Each window_idx block has num_nodes samples.
    
    # Correct way to split: split the *window indices* first
    window_indices_all = np.arange(dataset.num_windows)
    np.random.shuffle(window_indices_all) # Shuffle window indices if you want random windows in splits

    train_win_count = int(dataset.num_windows * (1 - val_ratio - test_ratio))
    val_win_count = int(dataset.num_windows * val_ratio)

    train_window_indices = window_indices_all[:train_win_count]
    val_window_indices = window_indices_all[train_win_count : train_win_count + val_win_count]
    test_window_indices = window_indices_all[train_win_count + val_win_count:]

    def get_dataset_indices(window_indices_subset):
        dataset_subset_indices = []
        for win_idx in window_indices_subset:
            for node_concept_idx in range(num_nodes): # For each of the 16 conceptual target nodes
                dataset_subset_indices.append(win_idx * num_nodes + node_concept_idx)
        return dataset_subset_indices

    train_indices = get_dataset_indices(train_window_indices)
    val_indices = get_dataset_indices(val_window_indices)
    test_indices = get_dataset_indices(test_window_indices)
    
    np.random.shuffle(train_indices) # Shuffle within the training set samples

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=100):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            if warmup_epochs == 0: return 1.0 # Avoid division by zero if no warmup
            return float(epoch + 1) / float(warmup_epochs) # epoch starts from 0
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, train_loader, val_loader, test_loader, optimizer, device,
               epochs=100, lambda_weight=0.4, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    scheduler = get_lr_scheduler(optimizer, warmup_epochs=5, max_epochs=epochs)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter, patience_limit = 0, 10

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_recon_loss_agg, epoch_pred_loss_agg, r2_agg, mae_agg, rmse_agg = 0, 0, 0, 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in progress_bar:
            batch = batch.to(device)
            predictions = model(batch)
            
            # loss_on_measured_nodes = F.mse_loss(predictions[batch.mask], batch.target[batch.mask])
            loss_on_unmeasured_nodes = F.mse_loss(predictions[~batch.mask], batch.target[~batch.mask]) 
            loss_on_unmeasured_nodes += 0.1 * torch.var(predictions[~batch.mask])
            
            # Add range-based loss term
            pred_range = torch.max(predictions[~batch.mask]) - torch.min(predictions[~batch.mask])
            target_range = torch.max(batch.target[~batch.mask]) - torch.min(batch.target[~batch.mask])
            range_loss = torch.abs(pred_range - target_range)
            loss_on_unmeasured_nodes += 0.05 * range_loss
            
            # Calculate R2 for unmeasured nodes
            y_true = batch.target[~batch.mask].detach()
            y_pred = predictions[~batch.mask].detach()
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero

            # Calculate MAE and RMSE for unmeasured nodes
            mae = torch.mean(torch.abs(y_true - y_pred))
            rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))

            # loss = lambda_weight * loss_on_measured_nodes + (1 - lambda_weight) * loss_on_unmeasured_nodes
            loss = loss_on_unmeasured_nodes

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            # epoch_recon_loss_agg += loss_on_measured_nodes.item()
            epoch_pred_loss_agg += loss_on_unmeasured_nodes.item()
            r2_agg += r2
            mae_agg += mae
            rmse_agg += rmse

            progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            # 'recon': f"{loss_on_measured_nodes.item():.4f}",
            'pred': f"{loss_on_unmeasured_nodes.item():.4f}",
            'R2': f"{r2.item():.3f}",
            'MAE': f"{mae.item():.3f}",
            'RMSE': f"{rmse.item():.3f}"
            })
        
        scheduler.step() 

        val_loss_epoch, val_pred_loss_epoch, val_r2_epoch, val_mae_epoch, val_rmse_epoch = evaluate_model(
            model, val_loader, device, lambda_weight
        )

        train_loss_epoch = epoch_loss / len(train_loader)
        # train_recon_epoch = epoch_recon_loss_agg / len(train_loader)
        train_pred_epoch = epoch_pred_loss_agg / len(train_loader)
        train_r2_epoch = r2_agg / len(train_loader)
        train_mae_epoch = mae_agg / len(train_loader)
        train_rmse_epoch = rmse_agg / len(train_loader)

        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Metrics: {train_loss_epoch:.6f} Pred: {train_pred_epoch:.6f} R2: {train_r2_epoch:.4f} MAE: {train_mae_epoch:.6f} RMSE: {train_rmse_epoch:.6f}")
        print(f"  Val Metrics: {val_loss_epoch:.6f} Pred: {val_pred_loss_epoch:.6f} R2: {val_r2_epoch:.4f} MAE: {val_mae_epoch:.6f} RMSE: {val_rmse_epoch:.6f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6e}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
            print(f"  New best model saved! Val loss: {val_loss_epoch:.6f}")
        else:
            patience_counter += 1
            print(f"  Validation didn't improve. Patience: {patience_counter}/{patience_limit}")

        # if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == epochs - 1 or val_loss_epoch < best_val_loss:
        visualize_predictions(model, train_loader, epoch, device, log_dir, num_batches=20, prefix='train')
        visualize_predictions(model, val_loader, epoch, device, log_dir, num_batches=8, prefix='val')
        plot_loss_curves(train_losses, val_losses, log_dir)

        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if os.path.exists(os.path.join(log_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pt")))
    else:
        print("No best model found, using last model state for testing.")


    test_loss, test_pred_loss, test_r2, test_mae, test_rmse = evaluate_model(
        model, test_loader, device, lambda_weight
    )
    print("\nTest Evaluation with Best Model:")
    print(f"  Test Loss: {test_loss:.6f} Pred: {test_pred_loss:.6f} R2: {test_r2:.4f} MAE: {test_mae:.6f} RMSE: {test_rmse:.6f}")

    return model, train_losses, val_losses

def evaluate_model(model, data_loader, device, lambda_weight=0.4):
    model.eval()
    epoch_loss_agg, epoch_pred_loss_agg, epoch_r2_agg, epoch_mae_agg, epoch_rmse_agg = 0, 0, 0, 0, 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            batch = batch.to(device)
            predictions = -model(batch)
            
            # loss_on_measured_nodes = F.mse_loss(predictions[batch.mask], batch.target[batch.mask])
            loss_on_unmeasured_nodes = F.mse_loss(predictions[~batch.mask], batch.target[~batch.mask])
            
            # Add range-based loss term
            pred_range = torch.max(predictions[~batch.mask]) - torch.min(predictions[~batch.mask])
            target_range = torch.max(batch.target[~batch.mask]) - torch.min(batch.target[~batch.mask])
            range_loss = torch.abs(pred_range - target_range)
            loss_on_unmeasured_nodes += 0.05 * range_loss
            
            # loss = lambda_weight * loss_on_measured_nodes + (1 - lambda_weight) * loss_on_unmeasured_nodes
            loss = loss_on_unmeasured_nodes

            # Calculate R2 for unmeasured nodes
            y_true = batch.target[~batch.mask].detach()
            y_pred = predictions[~batch.mask].detach()
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero

            # Calculate MAE and RMSE for unmeasured nodes
            mae = torch.mean(torch.abs(y_true - y_pred))
            rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))

            epoch_loss_agg += loss.item()
            # epoch_recon_loss_agg += loss_on_measured_nodes.item()
            epoch_pred_loss_agg += loss_on_unmeasured_nodes.item()
            epoch_r2_agg += r2
            epoch_mae_agg += mae
            epoch_rmse_agg += rmse
            
            progress_bar.set_postfix({
                'pred': f"{loss_on_unmeasured_nodes.item():.4f}",
                'R2': f"{r2:.4f}",
                'MAE': f"{mae:.4f}",
                'RMSE': f"{rmse:.4f}"
            })

    return (
        epoch_loss_agg / len(data_loader),
        epoch_pred_loss_agg / len(data_loader),
        epoch_r2_agg / len(data_loader),
        epoch_mae_agg / len(data_loader),
        epoch_rmse_agg / len(data_loader)
    )

def inverse_tanh_normalization(y_norm, scaling_factor=3.0):
    # Clip to avoid invalid values for arctanh
    y_norm = np.clip(y_norm, -1.999, 1.999)
    return scaling_factor * np.arctanh(y_norm / 2.0)

def visualize_predictions(model, data_loader, epoch, device, log_dir="logs", num_batches=20, prefix="val"):
    """Enhanced visualization function that plots actual vs. predicted for unmeasured sensors across multiple batches."""
    vis_dir = os.path.join(log_dir, "predictions")
    os.makedirs(vis_dir, exist_ok=True)
    
    debug_file = open(os.path.join(vis_dir, f"debug_viz_{epoch+1}.txt"), 'a')
    def log(msg):
        debug_file.write(f"{msg}\n")
        debug_file.flush()
    
    try:
        log(f"Starting visualization for epoch {epoch+1}")
        
        # Collect data from multiple batches
        all_batch_targets = []
        all_batch_predictions = []
        batch_mses = []  # Store MSE for each batch
        
        model.eval()
        with torch.no_grad():
            batch_count = 0
            for batch in data_loader:
                # if batch_count >= num_batches:
                #     break
                    
                batch = batch.to(device)
                predictions = -model(batch)
                
                # Calculate MSE for this batch (unmeasured nodes only)
                batch_targets_unmeasured = batch.target[model.unmeasured_nodes].cpu().numpy()
                batch_preds_unmeasured = predictions[model.unmeasured_nodes].cpu().numpy()
                batch_mse = np.mean((batch_targets_unmeasured - batch_preds_unmeasured) ** 2)
                
                # Store the batch data
                all_batch_targets.append(batch.target.cpu().numpy())
                all_batch_predictions.append(predictions.cpu().numpy())
                batch_mses.append(batch_mse)
                
                batch_count += 1
                log(f"Processed batch {batch_count}, MSE: {batch_mse:.6f}")
        
        if batch_count == 0:
            log("No batches processed, exiting visualization")
            return
            
        log(f"Collected {batch_count} batches for visualization")
        
        # Find batches with lowest MSE
        batch_mses = np.array(batch_mses)
        sorted_indices = np.argsort(batch_mses)  # Sort by MSE (ascending)
        best_batch_indices = sorted_indices[:8]  # Take 8 best batches
        
        log(f"Best batch MSEs: {batch_mses[best_batch_indices]}")
        log(f"Best batch indices: {best_batch_indices}")
        log(f"Worst batch MSE: {batch_mses[sorted_indices[-1]]:.6f}")
        log(f"Best batch MSE: {batch_mses[sorted_indices[0]]:.6f}")
        
        # 1. Individual unmeasured node predictions (using first batch for time series)
        first_batch_targets = all_batch_targets[0]
        first_batch_predictions = all_batch_predictions[0]
        
        log(f"First batch shapes - targets: {first_batch_targets.shape}, predictions: {first_batch_predictions.shape}")
        
        n_rows = 4
        n_cols = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
        axes = axes.flatten()
        
        # Plot all unmeasured nodes
        for i, node_idx in enumerate(model.unmeasured_nodes):
            ax = axes[i]
            # Inverse normalization for plotting
            target = inverse_tanh_normalization(first_batch_targets[node_idx])
            pred = inverse_tanh_normalization(first_batch_predictions[node_idx])

            log(f"Unmeasured Node {node_idx} - Target range: [{target.min():.4f}, {target.max():.4f}]")
            log(f"Unmeasured Node {node_idx} - Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
            
            # Plot
            ax.plot(target, 'b-', linewidth=2, label='Ground Truth')
            ax.plot(pred, 'r--', linewidth=2, label='Prediction')
            
            # Calculate metrics across all batches for this node (in original units)
            all_targets_for_node = []
            all_preds_for_node = []
            for batch_targets, batch_preds in zip(all_batch_targets, all_batch_predictions):
                all_targets_for_node.extend(batch_targets[node_idx::16])
                all_preds_for_node.extend(batch_preds[node_idx::16])
            all_targets_for_node = np.array(all_targets_for_node).flatten()
            all_preds_for_node = np.array(all_preds_for_node).flatten()
            # Inverse normalization for metrics
            all_targets_for_node = inverse_tanh_normalization(all_targets_for_node)
            all_preds_for_node = inverse_tanh_normalization(all_preds_for_node)

            mse = np.mean((all_targets_for_node - all_preds_for_node) ** 2)
            mae = np.mean(np.abs(all_targets_for_node - all_preds_for_node))
            if np.std(all_targets_for_node) > 0:
                r2 = 1 - np.sum((all_targets_for_node - all_preds_for_node) ** 2) / np.sum((all_targets_for_node - np.mean(all_targets_for_node)) ** 2)
            else:
                r2 = float('nan')
            correlation = np.corrcoef(all_targets_for_node, all_preds_for_node)[0, 1]
            
            # Set y limits with padding
            y_min = min(target.min(), pred.min()) - 0.1
            y_max = max(target.max(), pred.max()) + 0.1
            if abs(y_max - y_min) < 0.2:
                y_min -= 0.2
                y_max += 0.2
            ax.set_ylim([y_min, y_max])
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_title(f"Node {node_idx}\\nMSE: {mse:.6f}, R²: {r2:.3f}, Corr: {correlation:.3f}", fontsize=12)
            ax.set_xlabel("Time Steps", fontsize=10)
            ax.set_ylabel("Normalized Acceleration", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"{prefix}_unmeasured_nodes_epoch_{epoch+1:03d}.png")
        log(f"Saving individual nodes visualization to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Enhanced summary visualization with multiple metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect all unmeasured node data across all batches
        all_unmeasured_targets = []
        all_unmeasured_preds = []
        
        for batch_targets, batch_preds in zip(all_batch_targets, all_batch_predictions):
            batch_unmeasured_targets = batch_targets[model.unmeasured_nodes]
            batch_unmeasured_preds = batch_preds[model.unmeasured_nodes]
            all_unmeasured_targets.append(batch_unmeasured_targets)
            all_unmeasured_preds.append(batch_unmeasured_preds)
        
        all_unmeasured_targets = np.concatenate(all_unmeasured_targets, axis=0)
        all_unmeasured_preds = np.concatenate(all_unmeasured_preds, axis=0)
        all_unmeasured_targets = inverse_tanh_normalization(all_unmeasured_targets)
        all_unmeasured_preds = inverse_tanh_normalization(all_unmeasured_preds)
        all_targets_flat = all_unmeasured_targets.flatten()
        all_preds_flat = all_unmeasured_preds.flatten()
        
        # Overall scatter plot
        all_targets_flat = all_unmeasured_targets.flatten()
        all_preds_flat = all_unmeasured_preds.flatten()
        
        ax1.scatter(all_targets_flat, all_preds_flat, alpha=0.3, s=3, color='red')
        min_val = min(all_targets_flat.min(), all_preds_flat.min())
        max_val = max(all_targets_flat.max(), all_preds_flat.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        overall_mse = np.mean((all_targets_flat - all_preds_flat) ** 2)
        overall_r2 = 1 - np.sum((all_targets_flat - all_preds_flat) ** 2) / np.sum((all_targets_flat - np.mean(all_targets_flat)) ** 2)
        overall_corr = np.corrcoef(all_targets_flat, all_preds_flat)[0, 1]
        
        ax1.set_title(f"Overall Performance ({batch_count} batches)\\nMSE: {overall_mse:.6f}, R²: {overall_r2:.4f}, Corr: {overall_corr:.4f}", fontsize=12)
        ax1.set_xlabel("Ground Truth", fontsize=10)
        ax1.set_ylabel("Prediction", fontsize=10)
        ax1.grid(True, alpha=0.5)
        
        # Error distribution
        errors = all_targets_flat - all_preds_flat
        ax2.hist(errors, bins=50, alpha=0.7, color='red', density=True)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.8, linewidth=2)
        ax2.axvline(x=np.mean(errors), color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_title(f"Error Distribution\\nMean: {np.mean(errors):.6f}, Std: {np.std(errors):.6f}", fontsize=12)
        ax2.set_xlabel("Prediction Error", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.grid(True, alpha=0.5)
        
        # Per-node MSE
        node_mses = []
        for i, node_idx in enumerate(model.unmeasured_nodes):
            node_targets = all_unmeasured_targets[:, i].flatten()
            node_preds = all_unmeasured_preds[:, i].flatten()
            node_mse = np.mean((node_targets - node_preds) ** 2)
            node_mses.append(node_mse)
        
        ax3.bar(range(len(model.unmeasured_nodes)), node_mses, color='lightcoral', alpha=0.7)
        ax3.set_title("MSE by Unmeasured Node", fontsize=12)
        ax3.set_xlabel("Node Index", fontsize=10)
        ax3.set_ylabel("MSE", fontsize=10)
        ax3.set_xticks(range(len(model.unmeasured_nodes)))
        ax3.set_xticklabels([str(node) for node in model.unmeasured_nodes])
        ax3.grid(True, alpha=0.5)
        
        # Per-node R²
        node_r2s = []
        for i, node_idx in enumerate(model.unmeasured_nodes):
            node_targets = all_unmeasured_targets[:, i].flatten()
            node_preds = all_unmeasured_preds[:, i].flatten()
            if np.std(node_targets) > 0:
                node_r2 = 1 - np.sum((node_targets - node_preds) ** 2) / np.sum((node_targets - np.mean(node_targets)) ** 2)
            else:
                node_r2 = float('nan')
            node_r2s.append(node_r2)
        
        ax4.bar(range(len(model.unmeasured_nodes)), node_r2s, color='lightblue', alpha=0.7)
        ax4.set_title("R² by Unmeasured Node", fontsize=12)
        ax4.set_xlabel("Node Index", fontsize=10)
        ax4.set_ylabel("R²", fontsize=10)
        ax4.set_xticks(range(len(model.unmeasured_nodes)))
        ax4.set_xticklabels([str(node) for node in model.unmeasured_nodes])
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.5)
        
        plt.tight_layout()
        summary_path = os.path.join(vis_dir, f"{prefix}_enhanced_summary_epoch_{epoch+1:03d}.png")
        log(f"Saving enhanced summary to {summary_path}")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Multiple sample comparison (showing different unmeasured nodes across batches)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        num_samples_to_show = min(8, len(all_batch_targets))
        
        for sample_idx in range(num_samples_to_show):
            ax = axes[sample_idx]
            node_idx = model.unmeasured_nodes[sample_idx % len(model.unmeasured_nodes)]
            if sample_idx < len(all_batch_targets):
                target = inverse_tanh_normalization(all_batch_targets[sample_idx][node_idx])
                pred = inverse_tanh_normalization(all_batch_predictions[sample_idx][node_idx])
                
                ax.plot(target, 'b-', linewidth=2, label='Ground Truth')
                ax.plot(pred, 'r--', linewidth=2, label='Prediction')
                
                mse = np.mean((target - pred) ** 2)
                ax.set_title(f"Batch {sample_idx+1} - Node {node_idx}\\nMSE: {mse:.6f}", fontsize=10)
                ax.set_xlabel("Time Steps", fontsize=8)
                ax.set_ylabel("Normalized Acceleration", fontsize=8)
                if sample_idx == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.5)
            else:
                ax.set_visible(False)
        
        plt.tight_layout()
        samples_path = os.path.join(vis_dir, f"{prefix}_multiple_batches_epoch_{epoch+1:03d}.png")
        log(f"Saving multiple batches comparison to {samples_path}")
        plt.savefig(samples_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Measured vs Unmeasured context visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show measured nodes (ground truth - what the model uses as input)
        first_batch_targets = all_batch_targets[0]
        for i, node_idx in enumerate(model.measured_nodes):
            target = inverse_tanh_normalization(first_batch_targets[node_idx])
            ax1.plot(target, alpha=0.7, linewidth=1, label=f'Node {node_idx}' if i < 4 else '')
        
        ax1.set_title("Measured Nodes (Model Input)", fontsize=14)
        ax1.set_xlabel("Time Steps", fontsize=12)
        ax1.set_ylabel("Normalized Acceleration", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.5)
        
        # Show unmeasured nodes (predictions vs ground truth)
        first_batch_preds = all_batch_predictions[0]
        for i, node_idx in enumerate(model.unmeasured_nodes):
            target = inverse_tanh_normalization(first_batch_targets[node_idx])
            pred = inverse_tanh_normalization(first_batch_preds[node_idx])
            ax2.plot(target, '-', alpha=0.7, linewidth=1, label=f'GT {node_idx}' if i < 4 else '')
            ax2.plot(pred, '--', alpha=0.7, linewidth=1, label=f'Pred {node_idx}' if i < 4 else '')

        ax2.set_title("Unmeasured Nodes (Model Predictions)", fontsize=14)
        ax2.set_xlabel("Time Steps", fontsize=12)
        ax2.set_ylabel("Normalized Acceleration", fontsize=12)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.5)
        
        plt.tight_layout()
        context_path = os.path.join(vis_dir, f"{prefix}_context_comparison_epoch_{epoch+1:03d}.png")
        log(f"Saving context comparison to {context_path}")
        plt.savefig(context_path, dpi=300, bbox_inches='tight')
        plt.close()
          # 5. NEW: Best performing batches visualization (8 batches with lowest MSE)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        log(f"\\nCreating best batches visualization with {len(best_batch_indices)} batches")
        
        for plot_idx, best_batch_idx in enumerate(best_batch_indices):
            ax = axes[plot_idx]
            
            # Get the batch data
            batch_targets = all_batch_targets[best_batch_idx]
            batch_preds = all_batch_predictions[best_batch_idx]
            batch_mse = batch_mses[best_batch_idx]
            
            # Cycle through different unmeasured nodes for variety
            node_idx = model.unmeasured_nodes[plot_idx % len(model.unmeasured_nodes)]
            
            target = inverse_tanh_normalization(batch_targets[node_idx])
            pred = inverse_tanh_normalization(batch_preds[node_idx])
            
            ax.plot(target, 'b-', linewidth=2, label='Ground Truth')
            ax.plot(pred, 'r--', linewidth=2, label='Prediction')
            
            # Calculate node-specific MSE for this batch
            node_mse = np.mean((target - pred) ** 2)
            
            ax.set_title(f"Node {node_idx}\\nBatch MSE: {batch_mse:.6f}, Node MSE: {node_mse:.6f}", fontsize=10)
            ax.set_xlabel("Time Steps", fontsize=8)
            ax.set_ylabel("Normalized Acceleration", fontsize=8)
            if plot_idx == 0:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.5)
                    
        plt.suptitle(f"8 Best Performing Batches (Lowest MSE) - Epoch {epoch+1}", fontsize=14)
        plt.tight_layout()
        best_batches_path = os.path.join(vis_dir, f"{prefix}_best_batches_epoch_{epoch+1:03d}.png")
        log(f"Saving best batches visualization to {best_batches_path}")
        plt.savefig(best_batches_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. MSE distribution across all batches
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of batch MSEs
        ax1.hist(batch_mses, bins=min(15, len(batch_mses)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=np.mean(batch_mses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(batch_mses):.6f}')
        ax1.axvline(x=np.median(batch_mses), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(batch_mses):.6f}')
        ax1.set_title(f"Distribution of Batch MSEs ({len(batch_mses)} batches)", fontsize=12)
        ax1.set_xlabel("MSE", fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.5)
        
        # Line plot showing MSE progression
        ax2.plot(range(1, len(batch_mses)+1), batch_mses, 'o-', alpha=0.7, color='orange')
        ax2.axhline(y=np.mean(batch_mses), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(batch_mses):.6f}')
        
        # Highlight best batches
        for i, best_idx in enumerate(best_batch_indices):
            ax2.plot(best_idx+1, batch_mses[best_idx], 'ro', markersize=8, alpha=0.8)
            if i < 3:  # Only label first 3 to avoid clutter
                ax2.annotate(f'#{i+1}', (best_idx+1, batch_mses[best_idx]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_title("MSE Progression Across Batches", fontsize=12)
        ax2.set_xlabel("Batch Number", fontsize=10)
        ax2.set_ylabel("MSE", fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.5)
        
        plt.tight_layout()
        mse_dist_path = os.path.join(vis_dir, f"{prefix}_mse_distribution_epoch_{epoch+1:03d}.png")
        log(f"Saving MSE distribution to {mse_dist_path}")
        plt.savefig(mse_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log comprehensive statistics
        log(f"\\nComprehensive Statistics for Epoch {epoch+1}:")
        log(f"Overall MSE: {overall_mse:.6f}")
        log(f"Overall R²: {overall_r2:.6f}")
        log(f"Overall Correlation: {overall_corr:.6f}")
        log(f"Error Mean: {np.mean(errors):.6f}")
        log(f"Error Std: {np.std(errors):.6f}")
        log(f"Batch MSE Stats - Min: {batch_mses.min():.6f}, Max: {batch_mses.max():.6f}, Mean: {np.mean(batch_mses):.6f}")
        log(f"Processed {batch_count} batches with {len(all_targets_flat)} total predictions")
        
        log("Enhanced visualization completed successfully")
        
    except Exception as e:
        log(f"Error in visualization: {str(e)}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        debug_file.close()


def plot_loss_curves(train_losses, val_losses, log_dir="logs"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "loss_curves.png"))
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    outfile = "Graph/"+FILENAME+".pt"
    create_and_save_graph(filename="Cases/"+FILENAME+".mat", output_file=outfile)

    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if device.type == 'cuda': # For CUDA reproducibility
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log_dir = "logs_testing/" + FILENAME
    os.makedirs(log_dir, exist_ok=True)

    print("Loading graph data...")
    graph_data = torch.load(outfile, map_location=torch.device('cpu'), weights_only=False) # Load to CPU first
    # Manually move graph components to device later if needed, or let PyG handle it
    print(f"Loaded graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    print(f"Node feature dimension: {graph_data.num_node_features}")
    print(f"Edge feature dimension: {graph_data.num_edge_features}")
    print(f"Acceleration data shape: {graph_data.acceleration_data.shape}")

    print("Creating dataset...")
    dataset = WindowedAccelerationDataset(graph_data)

    val_ratio, test_ratio = 0.15, 0.15
    train_dataset, val_dataset, test_dataset = create_stratified_split(dataset, val_ratio, test_ratio)
    print(f"Split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    node_feature_dim = graph_data.num_node_features
    edge_feature_dim = graph_data.num_edge_features
    hidden_dim = 128
    dropout = 0.3 

    print("Initializing model...")
    model = EnhancedDGAR(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        window_size=48,
        history_size=5,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Adjusted lr and wd
    print(optimizer)

    print("Starting training...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, test_loader, optimizer, device,
        epochs=25, lambda_weight=0.2, log_dir=log_dir # lambda_weight for measured node loss
    )

    model_path = os.path.join(log_dir, "final_enhanced_dgar_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

if __name__ == "__main__":
    main()
