"""Upload a trained model folder to a PRIVATE HF model repo <user>/<name> (env HF_TOKEN)."""
import os, sys
from huggingface_hub import HfApi

folder, name = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = api.whoami()["name"] + "/" + name
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model",
                  ignore_patterns=["checkpoint-*/**"])
print("uploaded ->", repo, flush=True)
