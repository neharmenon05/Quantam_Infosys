from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from .processor import load_or_train_models, answer_query

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_PATH = os.path.join(BASE_DIR, "..", "policies.csv")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str | None = None):
    """Render main interface"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": None, "message": message},
    )


@app.post("/upload-csv")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """Upload CSV file and trigger quantum model training"""
    save_path = UPLOAD_PATH
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Train or reload the models
    load_or_train_models(save_path)

    return RedirectResponse(url="/?message=✅+Policies+uploaded+and+quantum+model+built+successfully!", status_code=303)


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str | None = Form(None),
    query: str | None = Form(None),
    top_k: int = Form(5),
):
    """Search top-k similar policies"""
    user_query = query if query else q

    if not user_query:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "message": "❌ Please enter a query before searching.",
            },
        )

    results = answer_query(user_query, top_k)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results, "message": None},
    )
