from pathlib import Path
import subprocess

BASE_DIR = Path("label_studio_docker")

FILES = {
    "docker-compose.yml": """services:
  label-studio:
    image: heartexlabs/label-studio:latest
    container_name: label-studio-local
    ports:
      - "8080:8080"
    env_file:
      - .env
    volumes:
      - ./mydata:/label-studio/data
      - ../data/lake:/label-studio/files/lake
    restart: unless-stopped
""",
    ".env": """LABEL_STUDIO_USERNAME=admin@example.com
LABEL_STUDIO_PASSWORD=admin123456
LABEL_STUDIO_PORT=8080

LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/
""",
}


def write_files():
    BASE_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "mydata").mkdir(exist_ok=True)
    (BASE_DIR / "myfiles").mkdir(exist_ok=True)

    for filename, content in FILES.items():
        path = BASE_DIR / filename
        path.write_text(content, encoding="utf-8")
        print(f"Created: {path}")


def run_docker_compose():
    subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=BASE_DIR,
        check=True,
    )


if __name__ == "__main__":
    write_files()

    print("\nStarting Label Studio...")
    run_docker_compose()

    print("\nDone.")
    print("Open: http://localhost:8080")
