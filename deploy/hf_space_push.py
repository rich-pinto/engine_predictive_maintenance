from huggingface_hub import HfApi
from pathlib import Path

HF_SPACE_REPO = "YOUR_USERNAME/engine-predictive-maintenance-space"
DEPLOY_DIR = Path("engine_predictive_maintenance/deploy")

api = HfApi()
api.upload_folder(
    folder_path=str(DEPLOY_DIR),
    repo_id=HF_SPACE_REPO,
    repo_type="space"
)

print("Deployment files uploaded to Hugging Face Space.")
