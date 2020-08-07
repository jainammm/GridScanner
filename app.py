from fastapi import FastAPI, Request, UploadFile, Response, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from predict.predict import predict

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "id": id})

@app.post("/scan/")
async def scan(file: UploadFile = File(...)):
    predict(file)
    return Response(status_code=200)