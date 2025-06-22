import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from tqdm import tqdm
from engine.evaluator import evaluate
from utils.monitoring import SpikeLogger
from torch.utils.tensorboard import SummaryWriter
from utils.visualizations import visualize_weights, visualize_batch_predictions
from utils.monitoring import log_from_monitors

def train(model, 
          train_loader, 
          val_loader, 
          optimizer, 
          device,
          loss_fn=None, 
          scaler=None, 
          epochs=10, 
          use_amp=False, 
          logger : SpikeLogger=None,
          save_intermediate=False):
    """
    Trains the model and evaluates on validation set each epoch.
    """
    model.to(device)
    model.train()


    def weighted_bce(preds, targets, pos_weight=5.0):
        weights = torch.where(targets == 1, pos_weight, 1.0)
        return F.binary_cross_entropy(preds, targets, weight=weights)

    if loss_fn is None:
        # pos_weight should be a 1-element tensor
        loss_fn = lambda preds, targets: weighted_bce(preds, targets, pos_weight=15.0)

    best_val_iou = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        for inputs, targets in loop:
            inputs = inputs.permute(1, 0, 2, 3, 4).to(device)   # [T, B, C, H, W]
            targets = targets.to(device) # [B, C, H, W]

            optimizer.zero_grad()

            # todo: use amp
            outputs = model(inputs)  # [B, C, H, W]      
            loss = loss_fn(outputs, targets) 
            loss.backward()
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            logger.log_scalar("Gradients/TotalL2", total_norm, epoch * len(train_loader) + total_batches)

            optimizer.step()
            
            total_batches += 1
            prev_avg_loss = running_loss / total_batches if total_batches > 1 else loss.item()
            running_loss += loss.item()
            avg_loss = running_loss / total_batches

            loop.set_postfix(train_loss=avg_loss)
         
            # log last batch
            if logger is not None and total_batches == len(train_loader) - 1:                
                log_from_monitors(model, logger, epoch)
            
            functional.reset_net(model)
            
            if loss.item() > (2.5 * prev_avg_loss):
                visualize_batch_predictions(inputs, targets, outputs, logger, epoch, title_tag="loss_spike")
            
            
            # clear monitors
            if hasattr(model, 'output_monitor') and model.output_monitor is not None:
                model.output_monitor.clear_recorded_data()
            if hasattr(model, 'v_monitor') and model.v_monitor is not None:
                model.v_monitor.clear_recorded_data()

        # Log training loss for the epoch
        train_loss = running_loss / total_batches
        logger.log_scalar("Loss/Train", train_loss, epoch)

        if logger is not None:
            log_from_monitors(model, logger, epoch)
            visualize_weights(
                model, 
                logger, 
                epoch,
                layer_logging_prefs={
                    "recurrent": "heatmap",      # recurrent layers → heatmap
                }
            )
        
        print("starting validation...")
        # Validation after each epoch
        val_loss, val_iou = evaluate(model, val_loader, device, loss_fn, use_amp)
        print(f"Validation - Loss: {val_loss:.4f}, Mean IoU: {val_iou:.4f}")


        # Log validation loss and IoU
        logger.log_scalar("Loss/Validation", val_loss, epoch)
        logger.log_scalar("IoU/Validation", val_iou, epoch)
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print("🟢 New best model found!")
            # Save model checkpoint
            if save_intermediate:
                torch.save(model.state_dict(), f"{logger.checkpoint_dir}/checkpoint_epoch_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"{logger.checkpoint_dir}/checkpoint_latest.pth")


        # Switch back to train mode for next epoch
        model.train()