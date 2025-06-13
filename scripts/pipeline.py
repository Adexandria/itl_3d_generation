from torch.utils.data import DataLoader, random_split
import torch
import torch.multiprocessing as mp
import argparse
import os
from torch.nn.utils.rnn import pad_sequence
from rnn import RNN
from pose_dataset import PoseDataset
from evaluation import evaluate_model
from train import train
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser(description="Train a Conditional VAE for pose generation")
    parser.add_argument('--out_dir', type=str, default='out_demo', help='Output directory to save the model and results')   
    parser.add_argument('--checkpoint', type=str, required= False, help='Path to the model checkpoint')
    parser.add_argument('--gloss_file', type=str, required=True, help='Json path to the gloss file containing keypoints')
    parser.add_argument('--is_train', action='store_true', help='Flag to indicate if training should be performed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cuda" or "cpu")')
    args = parser.parse_args()
   
    # Load the dataset
    dataset = PoseDataset(args.gloss_file,args.device)

    os.makedirs(args.out_dir, exist_ok=True)

    json_file = os.path.join(args.out_dir, 'dataset_info.json')
    ##os.(json_file, exist_ok=True)

    # Split the dataset into training and testing sets
    length = dataset.__len__()
    print(f"Dataset length: {length}")
    n_train = int(0.7 * length)
    n_validation = int(0.15 * length)
    n_test = length - n_train - n_validation

    train_ds, test_ds, validatio_ds = random_split(
        dataset,
        [n_train, n_test,n_validation],
        generator=torch.Generator().manual_seed(42))


    current_epoch = 1 
    
    # Initialize the  model
    model = RNN( gloss_dim=768,
    output_dim= 188,
    hidden_dim=128).to(args.device) # Adjust dimensions as needed

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if args.checkpoint is not None:
                checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                current_epoch = checkpoint['epoch'] + 1
                print(f"Loading from epoch {current_epoch-1}")
 
    if args.is_train:
        print("Loading training dataset...")
        print(f"Training dataset size: {len(train_ds)}")
        # Create DataLoader for training
        training_loader = DataLoader(train_ds,
                               batch_size=64,
                                 shuffle=True,  
                                 collate_fn=collate_fn,
                                 num_workers=4)
        validation_loader = DataLoader(validatio_ds,
                                 batch_size=64,
                                    shuffle=True, 
                                    collate_fn=collate_fn, 
                                    num_workers=4)
        # Initialize the training process
        trainer = train(model, args.out_dir, training_loader, validation_loader,
                        optimizer, num_epochs=20, device=args.device)
        checkpoint_path = trainer.run(current_epoch, json_file=json_file)

        print("Loading from the best checkpoint to evaluate the model")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        print(f"Loading from epoch {current_epoch-1}")
    
    print("Loading testing dataset...")
    print(f"Testing dataset size: {len(test_ds)}")
    test_loader = DataLoader(test_ds,
                        batch_size=64,   
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=4)
   
    # Evaluate the model
    print("Evaluating the model...")
    evaluation_loss = evaluate_model(model, test_loader, device=args.device)
    evaluation_loss.run(json_file=json_file, name = checkpoint_path)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None items
    
    if not batch:
        return None  # Return None if the batch is empty
    
    keypoints = [item[1] for item in batch]

    gloss_embeddings = torch.stack([item[0] for item in batch])

    lengths = torch.tensor([kp.shape[0] for kp in keypoints])

    keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0.0)  # Pad keypoints to the same length

    keypoints = torch.nan_to_num(keypoints, nan=0.0,posinf=0.0,neginf=0.0)  # Replace NaN and Inf values with 0.0
    
   # please confirm that the keypoints are in the correct format. the keypoins are correctly padded to the same length but my mask is not working properly
    mask = torch.arange(keypoints.size(1))[None, :] < lengths[:, None]

    mask = mask.unsqueeze(-1).repeat(1,1,keypoints.shape[2])  # Expand mask to match keypoints shape
    
    return gloss_embeddings, keypoints,mask
 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
