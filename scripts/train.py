import torch
import os
import json


class Train:
    def __init__(self, model, out_dir, train_ds, validation_ds, optimizer, num_epochs=5, device='cpu'):
        self.num_epochs = num_epochs
        self.train_loader = train_ds
        self.validation_loader = validation_ds
        self.out_dir = out_dir
        self.optimizer = optimizer
        self.model = model
        self.device = device

    def run(self, json_file, current_epoch=1):
        best_val_loss = float('inf')
        best_checkpoint = None
        patience = 0

        for epoch in range(current_epoch, self.num_epochs + current_epoch):
            self.model.train()
            total_loss = 0.0
            print("Starting training from epoch:", epoch)

            for gloss_embedding, keypoint, mask in self.train_loader:
                if gloss_embedding is None or keypoint is None or mask is None:
                    print("Skipping batch with None values")
                    continue

                # Repeat gloss embedding across time steps
                gloss_embedding = gloss_embedding.unsqueeze(1).repeat(1, keypoint.size(1), 1)

                # Move tensors to the specified device
                gloss_embedding = gloss_embedding.to(self.device)
                keypoint = keypoint.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                reconstructed = self.model(gloss_embedding)

                # Compute loss
                loss = self.model.train_loss(reconstructed, keypoint, mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

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

                    # Repeat gloss embedding across time steps
                    gloss_embedding = gloss_embedding.unsqueeze(1).repeat(1, keypoint.size(1), 1)
                    
                    # Move tensors to the specified device
                    gloss_embedding = gloss_embedding.to(self.device)
                    keypoint = keypoint.to(self.device)
                    mask = mask.to(self.device)

                    reconstructed = self.model(gloss_embedding)
                    val_loss = self.model.evaluation_loss(reconstructed, keypoint, mask)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(self.validation_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, total validation loss: {total_val_loss:.4f}, number of batches: {len(self.validation_loader)}")

            # Save best model checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
                save_path = os.path.join(self.out_dir, f"itl_3d_checkpoint_{epoch}.pth")
                best_checkpoint = save_path

                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                }, save_path)

                checkpoint_data = {
                    'checkpoint': save_path,
                    'epoch': epoch,
                    'loss': avg_loss,
                    'validation_loss': avg_val_loss,
                    'train_dataset': len(self.train_loader.dataset),
                    'validation_dataset': len(self.validation_loader.dataset)
                }

                if json_file:
                    with open(json_file, 'a') as f:
                        json.dump(checkpoint_data, f, indent=4)

                print(f"Model saved to {save_path} with validation loss: {avg_val_loss:.4f}")
            else:
                patience += 1
                print(f"No improvement in validation loss. Patience: {patience}")
                if patience >= 3:
                    print("Early stopping triggered.")
                    break

        print("Training complete")
        return best_checkpoint
