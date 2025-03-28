import os
import subprocess
import time
import requests
import psutil
import threading
import uuid
from flask import Flask, request, jsonify, render_template_string, g
from flask_cors import CORS
import logging

# ------------------- CONFIG -------------------
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

@app.before_request
def assign_request_id():
    g.request_id = uuid.uuid4().hex

UPLOAD_FOLDER = "uploads"
JD_UPLOAD_FOLDER = "jd_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JD_UPLOAD_FOLDER, exist_ok=True)

PYTHON_PATH = r"C:\Program Files\Python312\python.exe"

RESUME_APP = {
    "script_path": r"C:\Users\inc2956\OneDrive - Texila American University\Google_Ads_Campaign\AI_Agent\TatvaOne\Resume_Parser\script.py",
    "port": 8503,
    "name": "Resume Parser"
}

CERTIFICATE_APP = {
    "script_path": r"C:\Users\inc2956\OneDrive - Texila American University\Google_Ads_Campaign\AI_Agent\TatvaOne\Certificate_Application\Application.py",
    "port": 8504,
    "name": "Certificate Application"
}

FLASK_PORT = 8801

# ------------------- UTILITIES -------------------
def is_port_in_use(port):
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            return True
    return False

def start_streamlit(app_config):
    script_path = app_config["script_path"]
    port = app_config["port"]
    name = app_config["name"]

    if is_port_in_use(port):
        print(f"✅ {name} already running on port {port}")
        return

    print(f"🚀 Starting {name} on port {port}...")
    try:
        process = subprocess.Popen([
            PYTHON_PATH, "-m", "streamlit", "run", script_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        for _ in range(10):
            try:
                if requests.get(f"http://localhost:{port}").status_code == 200:
                    print(f"✅ {name} started at http://127.0.0.1:{port}")
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        stdout, stderr = process.communicate(timeout=10)
        print("Streamlit stdout:", stdout)
        print("Streamlit stderr:", stderr)

    except subprocess.TimeoutExpired:
        print(f"⚠️ {name} startup is taking too long...")
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")

def render_iframe(port, title):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; }}
            iframe {{ width: 100%; height: 90vh; border: none; }}
        </style>
    </head>
    <body>
        <h2>{title}</h2>
        <iframe src="http://localhost:{port}/?embed=true"></iframe>
    </body>
    </html>
    """
    return render_template_string(html)

# ------------------- FLASK ROUTES -------------------
@app.route("/")
def index():
    return """
    <h1>AI Agent Dashboard</h1>
    <ul>
        <li><a href="/resume_parser">Resume Parser App</a></li>
        <li><a href="/certificate_app">Certificate Application</a></li>
    </ul>
    """

@app.route("/resume_parser")
def resume_parser():
    start_streamlit(RESUME_APP)
    return render_iframe(RESUME_APP["port"], RESUME_APP["name"])

@app.route("/certificate_app")
def certificate_app():
    start_streamlit(CERTIFICATE_APP)
    return render_iframe(CERTIFICATE_APP["port"], CERTIFICATE_APP["name"])

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    print(f"📥 Uploaded resume saved: {save_path}")
    return jsonify({"message": "Resume uploaded successfully"}), 200

@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    save_path = os.path.join(JD_UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    print(f"📥 Uploaded JD saved: {save_path}")
    return jsonify({"message": "Job description uploaded successfully"}), 200

@app.route('/favicon.ico')
def favicon():
    return '', 204

# ------------------- FILE MANAGEMENT ENDPOINTS -------------------
@app.route('/clear_non_pdf_jd', methods=['POST'])
def clear_non_pdf_jd():
    removed = 0
    for file in os.listdir(JD_UPLOAD_FOLDER):
        if not file.lower().endswith(".pdf"):
            os.remove(os.path.join(JD_UPLOAD_FOLDER, file))
            removed += 1
    print(f"🗑️ Cleared {removed} non-PDF JD files.")
    return jsonify({"message": f"Removed {removed} non-PDF files.", "reset_merged": True}), 200

@app.route('/clear_all_pdfs', methods=['POST'])
def clear_all_pdfs():
    removed = 0
    for file in os.listdir(JD_UPLOAD_FOLDER):
        if file.lower().endswith(".pdf"):
            os.remove(os.path.join(JD_UPLOAD_FOLDER, file))
            removed += 1
    print(f"🗑️ Cleared {removed} PDF JD files.")
    return jsonify({"message": f"Removed {removed} PDF files.", "reset_merged": True}), 200

@app.route('/clear_all_jd_uploads', methods=['POST'])
def clear_all_jd_uploads():
    removed = 0
    for file in os.listdir(JD_UPLOAD_FOLDER):
        os.remove(os.path.join(JD_UPLOAD_FOLDER, file))
        removed += 1
    print(f"🗑️ Cleared ALL JD files ({removed} files).")
    return jsonify({"message": f"Removed all JD files ({removed} files).", "reset_merged": True}), 200

# ------------------- RUN FLASK -------------------
if __name__ == "__main__":
    #print(f"Flask running at http://0.0.0.0:{FLASK_PORT}")
    app.run(host='192.168.2.113', port=FLASK_PORT, debug=True)
