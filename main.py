from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Model Monitoring Reports")


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
def home():
    files = sorted(
        [
            p.name
            for p in STATIC_DIR.glob("*.html")
            if p.is_file()
        ]
    )

    if not files:
        return """
        <html>
          <head><title>Model Monitoring Reports</title></head>
          <body>
            <h1>Model Monitoring Reports</h1>
            <p>The app is running.</p>
            <p>No report files were found yet.</p>
            <p>Run <code>python generate_reports.py</code> locally, then deploy the generated <code>static/</code> folder.</p>
            <p>Health check: <a href="/health">/health</a></p>
          </body>
        </html>
        """

    links = "\n".join(
        f'<li><a href="/reports/{name}">{name}</a></li>'
        for name in files
    )

    return f"""
    <html>
      <head><title>Model Monitoring Reports</title></head>
      <body>
        <h1>Model Monitoring Reports</h1>
        <p>Health check: <a href="/health">/health</a></p>
        <ul>
          {links}
        </ul>
      </body>
    </html>
    """


app.mount("/reports", StaticFiles(directory=str(STATIC_DIR), html=True), name="reports")
