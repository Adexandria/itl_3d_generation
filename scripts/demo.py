from MeshGeneration import MeshGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train a Conditional VAE for pose generation")
    parser.add_argument('--out_dir', type=str, default='out_demo', help='Output directory to save the model and results')   
    parser.add_argument('--checkpoint', type=str, required= False, help='Path to the model checkpoint')
    parser.add_argument('--gloss_file', type=str, required=True, help='Json path to the gloss file containing keypoints')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--gloss', type=str, required=True, help='Gloss text to generate pose for')
    args = parser.parse_args()

    print("Running mesh generation with the following parameters:")
    # Create the mesh generator
    meshGenerator = MeshGenerator(
        gloss=args.gloss,  # Example gloss, replace with actual input
        checkpoint=args.checkpoint,
        gloss_file=args.gloss_file,
        out_folder=args.out_dir,
        save_mesh=args.save_mesh,  # Set to True if you want to save the mesh
        device=args.device
    )

    # Run the mesh generation
    meshGenerator.run()
    print("Mesh generation completed.")

if __name__ == "__main__":
        main()