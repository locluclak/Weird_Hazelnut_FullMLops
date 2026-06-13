import os
import zipfile
import subprocess
import sys
from pathlib import Path

# VM Configuration
VM_IP = "20.196.237.153"
VM_USER = "azureuser"
PEM_KEY = "ubuntu-vm_key.pem"
REMOTE_DIR = "/home/azureuser/weird-hazelnut"
ZIP_NAME = "deployment.zip"

EXCLUDED_DIR_NAMES = {
    ".git",
    "mlruns",
    "openvino_cache",
    "mlflow_data",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "hazelnut",
    "hazelnuts-15770",
    "hazelnut-crack-generated-by-ai"
}

EXCLUDED_FILES = {
    ZIP_NAME,
    PEM_KEY,
    "deployment.zip",
    "error.txt"
}

def create_deployment_zip(root_dir: Path, zip_path: Path):
    print("📦 Creating deployment zip archive...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in root_dir.rglob('*'):
            # Check if file_path is inside any excluded directory
            parts = file_path.relative_to(root_dir).parts
            if any(part in EXCLUDED_DIR_NAMES for part in parts):
                continue
            
            # Skip specific excluded files
            if file_path.name in EXCLUDED_FILES:
                continue
            
            # Skip directories that aren't leaf directories (zipfile adds them automatically)
            if file_path.is_dir():
                # We do want to ensure empty directories are created, like data/lake
                if file_path.name == "lake" and "data" in parts:
                    zip_file.writestr(str(file_path.relative_to(root_dir)) + "/", "")
                continue
                
            # Add file to zip
            archive_name = str(file_path.relative_to(root_dir))
            zip_file.write(file_path, archive_name)
    print(f"✅ Created deployment zip at: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")

def run_ssh_cmd(cmd: str):
    ssh_cmd = [
        "ssh", "-i", PEM_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"{VM_USER}@{VM_IP}",
        cmd
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"❌ SSH Command failed: {cmd}")
        print(f"Error output:\n{result.stderr}")
        return False
    stdout = result.stdout.strip()
    return stdout if stdout else True

def run_scp(local_path: str, remote_path: str):
    scp_cmd = [
        "scp", "-i", PEM_KEY,
        "-o", "StrictHostKeyChecking=no",
        local_path,
        f"{VM_USER}@{VM_IP}:{remote_path}"
    ]
    result = subprocess.run(scp_cmd)
    return result.returncode == 0

def main():
    root_dir = Path(__file__).resolve().parents[1]
    zip_path = root_dir / ZIP_NAME
    
    # 1. Create the deployment zip
    create_deployment_zip(root_dir, zip_path)
    
    # 2. Test connection
    print("🔌 Testing connection to Azure VM...")
    test_conn = run_ssh_cmd("echo 'Connection success!'")
    if not test_conn:
        print("❌ Could not connect to the VM. Please verify the IP and PEM key permissions.")
        sys.exit(1)
    print(f"✅ Connected: {test_conn}")
    
    # 3. Install Docker and unzip on the VM
    print("🛠️  Ensuring Docker, Docker Compose, and unzip are installed on the VM...")
    install_cmd = (
        "sudo apt-get update && "
        "sudo apt-get install -y unzip docker.io docker-compose-v2 && "
        "sudo usermod -aG docker azureuser"
    )
    print("Running apt-get and docker installation (this may take a minute)...")
    if not run_ssh_cmd(install_cmd):
        print("❌ Failed to check/install system dependencies on VM.")
        sys.exit(1)
    print("✅ System dependencies checked/installed.")
    
    # 4. Prepare remote directory
    print(f"📁 Preparing remote directory: {REMOTE_DIR}...")
    if not run_ssh_cmd(f"mkdir -p {REMOTE_DIR}"):
        print(f"❌ Failed to create remote directory {REMOTE_DIR} on VM.")
        sys.exit(1)
    
    # 5. SCP the zip and .env to the VM
    print("🚀 Uploading code archive and environment configuration...")
    if not run_scp(str(zip_path), REMOTE_DIR + "/deployment.zip"):
        print("❌ Failed to upload deployment.zip")
        sys.exit(1)
        
    env_path = root_dir / ".env"
    if env_path.exists():
        if not run_scp(str(env_path), REMOTE_DIR + "/.env"):
            print("❌ Failed to upload .env file")
            sys.exit(1)
    
    # 6. Unzip code on the VM
    print("🔓 Unzipping deployment package on the VM...")
    unzip_cmd = f"cd {REMOTE_DIR} && unzip -o deployment.zip && rm deployment.zip"
    if not run_ssh_cmd(unzip_cmd):
        print("❌ Failed to unzip deployment package on VM.")
        sys.exit(1)
    
    # 7. Initialize production model directory structure and copy models if not exist
    print("📂 Ensuring production model paths are populated...")
    model_prep_cmd = (
        f"cd {REMOTE_DIR} && "
        "mkdir -p models/anomaly_detector models/classifier && "
        "if [ ! -f models/anomaly_detector/model.xml ]; then "
        "  cp models/anomaly_detector_retrained/weights/openvino/model.xml models/anomaly_detector/model.xml && "
        "  cp models/anomaly_detector_retrained/weights/openvino/model.bin models/anomaly_detector/model.bin && "
        "  echo 'Anomaly detector models copied to production folder.'; "
        "fi && "
        "if [ ! -f models/classifier/model.onnx ]; then "
        "  cp models/classifier_retrained/model.onnx models/classifier/model.onnx && "
        "  cp models/classifier_retrained/model.onnx.data models/classifier/model.onnx.data && "
        "  cp models/classifier_retrained/meta.json models/classifier/meta.json && "
        "  echo 'Classifier models copied to production folder.'; "
        "fi"
    )
    if not run_ssh_cmd(model_prep_cmd):
        print("❌ Failed to prepare production models on VM.")
        sys.exit(1)
    
    # 8. Start docker compose on the VM
    print("🚢 Launching WeirdHazelnut Docker Compose stack on the VM...")
    compose_cmd = f"cd {REMOTE_DIR} && docker compose down && docker compose up -d --build"
    if not run_ssh_cmd(compose_cmd):
        print("❌ Failed to launch Docker Compose stack on VM.")
        sys.exit(1)
    print("✅ Docker Compose stack launched successfully!")
    
    # 9. Clean up local zip
    if zip_path.exists():
        zip_path.unlink()
        
    print("\n🎉 Triển khai thành công lên Azure VM!")
    print(f"URL API: http://{VM_IP}:8000")
    print(f"URL Label Studio: http://{VM_IP}:8080")
    print(f"URL MLflow: http://{VM_IP}:5000")
    print(f"URL Airflow: http://{VM_IP}:8088")
    print(f"URL MinIO Console: http://{VM_IP}:9001")
    print("\n👉 Hãy kết nối SSH vào VM để khởi tạo Label Studio Project và seed dữ liệu nếu là lần đầu chạy:")
    print(f"ssh -i {PEM_KEY} {VM_USER}@{VM_IP}")
    print(f"cd {REMOTE_DIR}")
    print("docker compose exec app python labeling/init_project.py")
    print("docker compose exec app python scripts/migrate_dataset_to_datalake.py --dataset-root data/hazelnut --dataset-name initial_hazelnut")

if __name__ == "__main__":
    main()
