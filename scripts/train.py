import torch
import os
import json
import datetime


class Train:
    def __init__(self, model, out_dir, train_ds, validation_ds, optimizer, scheduler=None, num_epochs=5, device='cpu'):
        self.num_epochs = num_epochs
        self.train_loader = train_ds
        self.validation_loader = validation_ds
        self.out_dir = out_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.device = device


    def run(self, json_file, current_epoch=1):
        best_val_loss = float('inf')
        best_checkpoint = None
        patience = 0

        for epoch in range(current_epoch, self.num_epochs):
            self.model.train()
            total_loss = 0.0
            print("Starting training from epoch:", epoch)

            for gloss_embedding, keypoint, mask in self.train_loader:
                if gloss_embedding is None or keypoint is None or mask is None:
                    print("Skipping batch with None values")
                    continue

                # Debug: Print text embedding statistics
                if torch.isnan(gloss_embedding).any():
                    print("WARNING: NaN detected in gloss_embedding")
                
                # Check text diversity across batch
                batch_text_std = torch.std(gloss_embedding, dim=0).mean()
                if batch_text_std < 1e-6:
                    print(f"WARNING: Very low text variation in batch. Std: {batch_text_std:.8f}")
                    print("This could cause identical outputs for different texts!")
                
                # Move tensors to the specified device first
                gloss_embedding = gloss_embedding.to(self.device)
                keypoint = keypoint.to(self.device)
                mask = mask.to(self.device)
                
                # Add input noise for regularization
                # Progressive noise reduction: high noise early, low noise later
                progress = epoch / self.num_epochs
                noise_factor = max(0.3, 1.0 - progress * 0.7)  # Keep higher noise throughout training
                    
                # Add stronger noise to text embeddings to prevent mode collapse
                text_noise_std = 0.02 * noise_factor  # Doubled the base noise
                text_noise = torch.randn_like(gloss_embedding) * text_noise_std
                gloss_embedding = gloss_embedding + text_noise
                    
                # Add noise to keypoints to improve robustness
                pose_noise_std = 0.01 * noise_factor  # Doubled pose noise too
                pose_noise = torch.randn_like(keypoint) * pose_noise_std
                keypoint = keypoint + pose_noise

                # Forward pass
                reconstructed, p_tf = self.model(gloss_embedding,keypoint,epoch,self.num_epochs)
                
                # Compute main loss
                main_loss = self.model.train_loss(reconstructed, keypoint, mask)
                
                # Debug: Monitor losses and teacher forcing
                if epoch % 5 == 0:  # Print every 5 epochs
                    print(f"Epoch {epoch}: Teacher forcing prob: {p_tf:.4f}")
                    print(f"Output std: {torch.std(reconstructed):.6f}")
                    print(f"Input text std: {torch.std(keypoint):.6f}")
                
                self.optimizer.zero_grad()
                main_loss.backward()
                
                # Add gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += main_loss.item()  # Use main_loss for averaging

            avg_loss = total_loss / len(self.train_loader)
            print(f"Average loss for epoch {epoch}: {avg_loss:.4f}, total loss: {total_loss:.4f}, number of batches: {len(self.train_loader)}")

            # Validation phase
            print("Starting validation...")
            self.model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for gloss_embedding, keypoint, mask in self.validation_loader:
                    if gloss_embedding is None or keypoint is None or mask is None:
                        print("Skipping batch with None values")
                        continue
                    
                    # Move tensors to the specified device
                    gloss_embedding = gloss_embedding.to(self.device)
                    keypoint = keypoint.to(self.device)
                    mask = mask.to(self.device)

                    reconstructed = self.model.forward_autoregression(gloss_embedding)
                    val_loss = self.model.evaluation_loss(reconstructed, keypoint, mask)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(self.validation_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, total validation loss: {total_val_loss:.4f}, number of batches: {len(self.validation_loader)}")

            # Step the scheduler with validation loss
            if self.scheduler:
                self.scheduler.step(avg_val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")

            # Save model checkpoint
            save_path = os.path.join(self.out_dir, f"itl_3d_checkpoint_{epoch}.pth")
            
            # Save current epoch checkpoint
            torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'val_loss': avg_val_loss,
                }, save_path)
            print(f"Model saved to {save_path} with validation loss: {avg_val_loss:.4f}")
            
            # Update best checkpoint if this is the best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint = save_path
                patience = 0  # Reset patience counter
                # Also save as best model
                best_path = os.path.join(self.out_dir, "best_model.pth")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'val_loss': avg_val_loss,
                }, best_path)
                print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
            else:
                patience += 1
                print(f"No improvement for {patience} epochs")
                
                # Early stopping
                if patience >= 50:  # Stop if no improvement for 5 epochs
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break    

            model_data = {
                'checkpoint': save_path,
                'epoch': epoch,
                'train_loss': avg_loss,
                'probability_tf': p_tf,
                'validation_loss': avg_val_loss,
                'train_dataset': len(self.train_loader.dataset),
                'validation_dataset': len(self.validation_loader.dataset),
                'timestamp': datetime.datetime.now().isoformat()
            }

            if json_file:
                with open(json_file, 'a') as f:
                    json.dump(model_data, f, indent=4)

            
        print("Training complete")
        return best_checkpoint
