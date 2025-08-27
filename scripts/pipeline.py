from torch.utils.data import DataLoader, random_split
import torch
import os
import argparse
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from rnn import RNN
from pose_dataset import PoseDataset
from evaluation import evaluate_model
from train import Train
import torch.optim as optim
import random

def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None items
    if not batch:
        return None
    
    keypoints = [item[1] for item in batch]
    gloss_embeddings = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([kp.shape[0] for kp in keypoints])

    keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0.0)
    keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

    mask = torch.arange(keypoints.size(1))[None, :] < lengths[:, None]
    mask = mask.unsqueeze(-1).repeat(1, 1, keypoints.shape[2])

    gloss_embeddings = gloss_embeddings.unsqueeze(1).repeat(1, keypoints.size(1), 1)
    return gloss_embeddings, keypoints, mask

def main():
    parser = argparse.ArgumentParser(description="Train a RNN model pose generation")
    parser.add_argument('--out_dir', type=str, default='out_demo', help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=False, help='Checkpoint path')
    parser.add_argument('--gloss_file', type=str, required=True, help='Gloss file path')
    parser.add_argument('--is_train', action='store_true', help='Enable training')
    args = parser.parse_args() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    else:
        torch.manual_seed(42)

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = PoseDataset(args.gloss_file, device)

    length = len(dataset)
    print(f"Dataset length: {length}")

    n_train = int(0.7 * length)
    n_val = int(0.15 * length)
    n_test = length - n_train - n_val

    train_ds, test_ds, val_ds = random_split(
        dataset,
        [n_train, n_test, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    model = RNN(gloss_dim=768, output_dim=188, hidden_dim=128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    current_epoch = 1
    checkpoint_path = None

    json_file = os.path.join(args.out_dir, 'dataset_info.json')
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {current_epoch - 1}")
        checkpoint_path = args.checkpoint
    
    if args.is_train:
        print("Training size:", len(train_ds))
        # Adjust batch size based on dataset size and memory constraints
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)

        trainer = Train(model, args.out_dir, train_loader, val_loader, optimizer, scheduler, num_epochs=201, device=device)
        checkpoint_path = trainer.run(current_epoch=current_epoch, json_file=json_file)

        # Load best model for testing
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("Testing size:", len(test_ds))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)

    print("Evaluating...")
    evaluation_loss = evaluate_model(model, test_loader, device=device)
    evaluation_loss.run(json_file=json_file,name=checkpoint_path)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    random.seed(42)

    main()
