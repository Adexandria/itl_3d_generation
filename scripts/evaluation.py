import torch
import json
class evaluate_model:
    def __init__(self, model, loader, device='cpu'):
        self.device = device
        self.loader = loader
        self.model = model
    
    def run(self,json_file,name):
        num_samples = 0
        total_loss= 0.0
        self.model.eval()
        print(f"Starting evaluation with the model from {name}")
        with torch.no_grad():
            for gloss_embedding, keypoint,mask in self.loader:
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
                num_samples += 1
                avg_loss = self.model.evaluation_loss(reconstructed, keypoint,mask)
                
                total_loss += avg_loss.item()
        
        avg_loss = total_loss / num_samples
        total_loss = 100 * avg_loss
        print(f"Total loss: {total_loss:.2f}%")
        print(f"Total samples: {num_samples}, Evaluation complete.") 
        checkpoint_data = {
                        'checkpoint': name,
                        'total_loss': avg_loss,
                        'num_samples': num_samples
                    }
        if json_file:
            with open(json_file, 'a') as f:
                json.dump(checkpoint_data, f, indent=4)
