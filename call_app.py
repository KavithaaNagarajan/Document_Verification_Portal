import subprocess
import time
import requests
import psutil
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# ✅ Define paths
PYTHON_PATH = r"C:\Users\inc3061\OneDrive - Texila American University\Documents\Resumepath\Scripts\python.exe"  # Update if necessary
SCRIPT_PATH = r"C:\Users\inc3061\OneDrive - Texila American University\Python\Certificate_Application\Application.py"
STREAMLIT_PORT = 8501  # Change if needed

def is_streamlit_running():
    """Check if Streamlit is already running on the given port."""
    try:
        response = requests.get(f"http://127.0.0.1:{STREAMLIT_PORT}", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def stop_existing_streamlit():
    """Stop any existing Streamlit process before starting a new one."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.name().lower() == "python.exe" and any("streamlit" in cmd for cmd in proc.cmdline()):
                print(f"🛑 Stopping existing Streamlit process (PID {proc.pid})...")
                proc.terminate()
                proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def start_streamlit():
    """Start Streamlit and print logs to diagnose issues."""
    if is_streamlit_running():
        print(f"✅ Streamlit is already running at http://127.0.0.1:{STREAMLIT_PORT}")
        return
    
    stop_existing_streamlit()

    print(f"🚀 Starting Streamlit on port {STREAMLIT_PORT}...")
    process = subprocess.Popen(
        [PYTHON_PATH, "-m", "streamlit", "run", SCRIPT_PATH,
         "--server.port", str(STREAMLIT_PORT), "--server.headless", "true",
         "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # ✅ Wait up to 30 seconds for Streamlit to start
    for _ in range(30):
        if is_streamlit_running():
            print(f"✅ Streamlit started successfully at http://127.0.0.1:{STREAMLIT_PORT}")
            return
        time.sleep(1)

    # ❌ If Streamlit does not start, print the error logs
    print("❌ Streamlit did not start properly!")
    stdout, stderr = process.communicate()
    print("📜 Streamlit Logs:")
    print(stdout)
    print(stderr)

@app.route("/certificate_app", methods=["GET"])
def certificate_app():
    """Embed Streamlit app inside Flask using an iframe."""
    start_streamlit()  # Ensure Streamlit is running

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Certificate Application</title>
        <style>
            body {{ text-align: center; font-family: Arial, sans-serif; }}
            iframe {{ width: 100%; height: 90vh; border: none; }}
        </style>
    </head>
    <body>
        <h2>Certificate Application</h2>
        <iframe src="http://127.0.0.1:{STREAMLIT_PORT}/?embed=true"></iframe>
    </body>
    </html>
    """
    return render_template_string(html_content)

# ✅ Fix 404 errors for favicon.ico
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content (fixes 404 log spam)

if __name__ == "__main__":
    print("✅ Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)