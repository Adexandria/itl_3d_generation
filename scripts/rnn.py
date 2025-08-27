from torch import nn
import torch
import math
import random
class RNN(nn.Module):
    def __init__(self, gloss_dim, output_dim, hidden_dim):
        super().__init__()
        self.pose_dim = output_dim
        self.initial_pose = nn.Parameter(torch.zeros((1, 1, output_dim)))
        self.encoder = nn.GRU(
            input_size=gloss_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,    
            num_layers=2
        )  
        self.text_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.encoder_to_decoder = nn.Linear(hidden_dim * 2, hidden_dim)
        # Start token for pose
        self.decoder = nn.GRU(
            input_size= self.pose_dim + hidden_dim * 2,  # Concatenate pose and hidden state
            hidden_size=hidden_dim,
            batch_first=True
        )
       
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=0.1, batch_first=True)
        self.attention_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)


    # for batch forward pass for validation and testing
   

    # for batch forward pass for training
    def forward(self, gloss_emb, ground_truth, epoch, total_epochs): 
        if torch.isnan(gloss_emb).any():
            print("NaN detected in input!")

        # Get the number of time steps
        B,T,_= gloss_emb.shape
        
        text,text_h = self.encoder(gloss_emb) #(D*2,B,H) Get the last hidden state from the encoder
        encoder_final = torch.cat((text_h[-2], text_h[-1]), dim=-1)  # Concatenate [B, H*2] from both directions
        hidden = self.encoder_to_decoder(encoder_final).unsqueeze(0)

        projected_text = self.text_projection(text)  # Project text features to hidden_dim

        progress = epoch / total_epochs  # Progress from 0 to 1
        k = 10  # Steepness parameter - higher values make the transition sharper
        p_tf = 1.0 / (1.0 + math.exp(k * (progress - 0.5)))
        
     
        prev_pose = self.initial_pose.repeat(B, 1, 1).to(text_h.device)  # Initialize pose with zeros [B,1, output_dim]
        prev_pose = prev_pose + torch.randn_like(prev_pose) * 0.01  # Add small noise to initial pose
        poses = []

    
        for i in range(T):
            input  = torch.cat([prev_pose, text[:, i:i+1, :]], dim=-1)  # Concatenate pose and hidden state [B, 1, output_dim + H*2]
            output,hidden = self.decoder(input,hidden)  # [B, 1, H]
            hidden_query = hidden.transpose(0, 1)
            projected_text_i = projected_text[:, i:i+1, :]  # Get the text projection for the current time step
            context, _ = self.attention(
            hidden_query, 
            projected_text_i, 
            projected_text_i
            )
            combined = torch.cat([output, context], dim=-1)  # Concatenate output and context [B, 1, H + H*2]
            projected = self.attention_projection(combined)  # Project to hidden_dim # Project to hidden_dim
            final_output = projected + output
            normalized_output = self.layer_norm(final_output) 
            next_pose = self.fc(normalized_output)  # [B, 1, output_dim] 
            poses.append(next_pose)  # Append the predicted pose
            
            if random.random() < p_tf:
                prev_pose = ground_truth[:, i:i+1, :]  
            else:
                prev_pose = next_pose
        
        output = torch.cat(poses, dim=1)  # Concatenate all predicted poses [B, T, output_dim]
        return output,p_tf
    
    def forward_autoregression(self, gloss_emb): 
        if torch.isnan(gloss_emb).any():
            print("NaN detected in input!")

        T = gloss_emb.size(1)  # Get the number of time steps

        output = self.forward_single(gloss_emb, T)  # Call the single forward pass method
           
        return output
    
    def forward_single(self, gloss_emb, frame_length):
        if torch.isnan(gloss_emb).any():
            print("NaN detected in input!")

        text,text_h = self.encoder(gloss_emb) #(D*2,B,H) Get the last hidden state from the encoder
        encoder_final = torch.cat((text_h[-2], text_h[-1]), dim=-1)  # Concatenate [B, H*2] from both directions
        hidden = self.encoder_to_decoder(encoder_final).unsqueeze(0) 

        projected_text = self.text_projection(text)  # Project text features to hidden_dim
        pose = []
        t_pose = self.initial_pose.repeat(gloss_emb.size(0), 1, 1).to(text_h.device)  # Initialize pose with zeros [B,1, output_dim]
        t_pose = t_pose + torch.randn_like(t_pose) * 0.01  # Add small noise to initial pose

        for i in range(frame_length):
            input  = torch.cat([t_pose, text[:, i:i+1, :]], dim=-1)  # Concatenate pose and hidden state [B, 1, output_dim + H*2]
            output,hidden = self.decoder(input,hidden)  # [B, 1, H]
            hidden_query = hidden.transpose(0, 1)
            projected_text_i = projected_text[:, i:i+1, :]  # Get the text projection for the current time step
            context, _ = self.attention(
            hidden_query, 
            projected_text_i, 
            projected_text_i
            )
            combined = torch.cat([output, context], dim=-1)  # Concatenate output and context [B, 1, H + H*2]
            projected = self.attention_projection(combined)  # Project to hidden_dim # Project to hidden_dim
            final_output = projected + output
            normalized_output = self.layer_norm(final_output)
            next_pose = self.fc(normalized_output)  # [B, 1, output_dim]
            pose.append(next_pose)  # Append the predicted pose
            t_pose = next_pose # Update t_pose for the next iteration
        
        out_pose = torch.cat(pose, dim=1)  # Concatenate all predicted poses [B, T, output_dim]
        return out_pose

    def train_loss(self,input, target, mask):
        if torch.isnan(input).any():
            print("NaN detected in input!")
        if torch.isnan(target).any():
            print("NaN detected in target!")
        if torch.isnan(mask).any():
            print("NaN detected in mask!")
             
        # Compute the mean squared error loss between the reconstructed output and the input data
        mse_loss = nn.MSELoss(reduction="none").to(input.device)  # Use 'none' reduction to compute element-wise loss

        loss = mse_loss(input, target) # Print the first element of the mask for debugging
        mask_loss = (loss * mask.float()).sum()  # Apply the mask to the loss
        non_zero_elements = mask.sum()
        mse_loss_val = mask_loss / non_zero_elements 
        
        return mse_loss_val 

    def evaluation_loss(self, input, target, mask):
        # Reconstruct the input from the output    
        reconstruct_joints = input[:,:,20:185]
        ground_truth_joints = target[:,:,20:185]

        # reshape joints to (B, T, 55, 3)
        num_joints = reconstruct_joints.shape[2] // 3
        
        reconstruct_joints = reconstruct_joints.view(input.shape[0], input.shape[1], num_joints, 3)
        ground_truth_joints = ground_truth_joints.view(target.shape[0], target.shape[1], num_joints, 3)

        
        joint_mask = mask[:,:,20:185]  # Extract the mask for the joint positions
        joint_mask = joint_mask.view(mask.shape[0], mask.shape[1], num_joints,3)  # Reshape mask to match joint dimensions

        collapsed_mask = joint_mask.sum(dim=-1).float()

        # Calculate the mean per joint position error

        loss = torch.norm(reconstruct_joints - ground_truth_joints,dim=-1)  # L2 norm along the joint dimension
        
        loss = (loss * collapsed_mask).sum()  # Apply the mask to the loss

        non_zero_elements = joint_mask.sum()

        mpjpe = loss / non_zero_elements

        return mpjpe