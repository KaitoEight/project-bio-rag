import os
import argparse
from huggingface_hub import snapshot_download

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen/Qwen2.5-3B-Instruct",
    "openai/clip-vit-base-patch16",
    "google/owlvit-base-patch32",
    "5CD-AI/Vintern-1B-v2"
]

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models for offline use.")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the models")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    hf_token = os.getenv("HF_TOKEN")

    for model_id in MODELS:
        print(f"Downloading {model_id}...")
        model_path = os.path.join(save_dir, model_id.split("/")[-1])
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,  # Download actual files so they can be copied to Google Drive
                token=hf_token
            )
            print(f"Successfully downloaded {model_id} to {model_path}\n")
        except Exception as e:
            print(f"Failed to download {model_id}: {e}\n")

if __name__ == "__main__":
    main()
