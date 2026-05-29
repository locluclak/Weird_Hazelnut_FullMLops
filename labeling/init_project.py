import logging
import os
import sys
import json
from urllib import request, parse, error

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config
from label_studio_sdk.client import LabelStudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_config_path():
    path = os.getenv("CONFIG_PATH", "config.yaml")
    logger.info(f"Using config path: {path}")
    return path

def load_ls_config():
    try:
        cfg = load_config()
        return cfg.get("label_studio", {})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

CONFIG_PATH = get_config_path()
ls_cfg = load_ls_config()

LABEL_STUDIO_URL = ls_cfg.get("url", "http://localhost:8080")
API_KEY = ls_cfg.get("api_key", "PASTE_YOUR_API_KEY_HERE")
print(f"\nAPI Key: {API_KEY}")
logger.info(f"Connecting to Label Studio at: {LABEL_STUDIO_URL}")

PROJECT_TITLE = "WeirdHazelnut Quality Audit"
LOCAL_STORAGE_TITLE = "WeirdHazelnut Lake Images"
LOCAL_STORAGE_PATH = "/label-studio/files/lake"
_ACCESS_TOKEN = None

LABEL_CONFIG = """
<View>
  <Header value="Hazelnut Quality Control"/>
  <Image name="image" value="$image"/>
  
  <View style="display: flex; justify-content: space-between">
    <View style="width: 45%">
      <Header value="1. Initial Assessment"/>
      <Choices name="sentiment" toName="image" showInLine="true">
        <Choice value="Normal" background="green"/>
        <Choice value="Anomaly" background="red"/>
      </Choices>
    </View>
    
    <View style="width: 45%">
      <Header value="2. Defect Classification"/>
      <Choices name="label" toName="image" visibleWhen="choice-selected" whenTagName="sentiment" whenChoiceValue="Anomaly">
        <Choice value="crack"/>
        <Choice value="cut"/>
        <Choice value="hole"/>
        <Choice value="print"/>
      </Choices>
    </View>
  </View>
  
  <Header value="Model Guidance"/>
  <Text name="score_info" value="Anomaly Score: $anomaly_score"/>
</View>
"""

def update_config_project_id(project_id):
    try:
        with open(CONFIG_PATH, "r") as f:
            lines = f.readlines()
        
        with open(CONFIG_PATH, "w") as f:
            for line in lines:
                if "project_id:" in line:
                    indent = line.split("project_id:")[0]
                    f.write(f"{indent}project_id: {project_id}\n")
                else:
                    f.write(line)
        logger.info(f"Updated {CONFIG_PATH} with new project_id: {project_id}")
    except Exception as e:
        logger.error(f"Failed to update {CONFIG_PATH}: {e}")

def get_authorization_header():
    global _ACCESS_TOKEN

    if API_KEY.count(".") != 2:
        return f"Token {API_KEY}"

    if _ACCESS_TOKEN:
        return f"Bearer {_ACCESS_TOKEN}"

    url = LABEL_STUDIO_URL.rstrip("/") + "/api/token/refresh"
    payload = {"refresh": API_KEY}
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            _ACCESS_TOKEN = body["access"]
            return f"Bearer {_ACCESS_TOKEN}"
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with {e.code}: {body}") from e

def ls_api(method, path, payload=None, query=None):
    url = LABEL_STUDIO_URL.rstrip("/") + path
    if query:
        url += "?" + parse.urlencode(query)

    data = None
    headers = {
        "Authorization": get_authorization_header(),
        "Content-Type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {e.code}: {body}") from e

def ensure_local_file_storage(project_id):
    storages = ls_api("GET", "/api/storages/localfiles/", query={"project": project_id}) or []
    for storage in storages:
        if storage.get("path") == LOCAL_STORAGE_PATH:
            logger.info(
                "Local file storage already configured for project %s: %s",
                project_id,
                LOCAL_STORAGE_PATH,
            )
            return storage

    payload = {
        "project": project_id,
        "title": LOCAL_STORAGE_TITLE,
        "description": "Serves routed hazelnut images from data/lake.",
        "path": LOCAL_STORAGE_PATH,
        "recursive_scan": True,
        "regex_filter": ".*\\.(png|jpg|jpeg)$",
        "use_blob_urls": False,
    }
    storage = ls_api("POST", "/api/storages/localfiles/", payload=payload)
    logger.info(
        "Created local file storage for project %s at %s",
        project_id,
        LOCAL_STORAGE_PATH,
    )
    return storage

def project_exists(project_id):
    try:
        ls_api("GET", f"/api/projects/{project_id}")
        return True
    except RuntimeError as e:
        if " failed with 404:" in str(e):
            return False
        raise

def initialize_project():
    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        logger.error("Please set your API_KEY in config.yaml or config.docker.yaml.")
        return

    try:
        ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

        project_id = ls_cfg.get("project_id")
        if project_id:
            project_id = int(project_id)
            if project_exists(project_id):
                logger.info(f"Using existing project ID from config: {project_id}")
            else:
                logger.warning(
                    "Configured project_id %s does not exist in Label Studio. Creating a new project.",
                    project_id,
                )
                project_id = None

        if not project_id:
            project = ls.projects.create(
                title=PROJECT_TITLE,
                label_config=LABEL_CONFIG,
                description="Human-in-the-loop audit for uncertain hazelnut detections."
            )
            project_id = project.id
            logger.info(f"Successfully created project ID: {project_id}")
            update_config_project_id(project_id)

        ensure_local_file_storage(project_id)
        logger.info(f"Access it at: {LABEL_STUDIO_URL}/projects/{project_id}")
        
    except Exception as e:
        logger.error(f"Error initializing project: {e}")
        logger.error(f"Make sure Label Studio is running at {LABEL_STUDIO_URL} and the API key is valid.")

if __name__ == "__main__":
    initialize_project()
