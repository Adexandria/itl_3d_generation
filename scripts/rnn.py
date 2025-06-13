from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
class RNN(nn.Module):
    def __init__(self, gloss_dim, output_dim, hidden_dim):
        super().__init__()
        self.pose_dim = output_dim
        self.initial_pose = torch.zeros((1, 1, output_dim))
        self.encoder = nn.GRU(
            input_size=gloss_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,    
            num_layers=2
        )  
        # Start token for pose
        self.decoder = nn.GRU(
            input_size= self.pose_dim + hidden_dim * 2,  # Concatenate pose and hidden state
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, gloss_emb): 
        if torch.isnan(gloss_emb).any():
            print("NaN detected in input!")

        T = gloss_emb.size(1)  # Get the number of time steps
        _,h = self.encoder(gloss_emb) #(D*2,B,H) Get the last hidden state from the encoder
        h = torch.cat((h[-2], h[-1]), dim=-1)  # Concatenate [B, H*2] from both directions
        h = h.unsqueeze(1).repeat(1, T, 1)  # [B,T,H] Repeat the hidden state for each time step
        hidden = None
        pose = []
        t_pose = self.initial_pose.repeat(gloss_emb.size(0), 1, 1)  # Initialize pose with zeros [B,1, output_dim]

        for i in range(T):
            input  = torch.cat([t_pose, h[:, i:i+1, :]], dim=-1)  # Concatenate pose and hidden state [B, 1, output_dim + H*2]
            output,hidden = self.decoder(input,hidden)  # [B, 1, H]
            next_pose = self.fc(output)  # [B, 1, output_dim]
            pose.append(next_pose)  # Append the predicted pose
            t_pose = next_pose # Update t_pose for the next iteration
           
        
        out_pose = torch.cat(pose, dim=1)  # Concatenate all predicted poses [B, T, output_dim]
        return out_pose
    

    def train_loss(self,input, target,mask):
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