# GridScanner

## Overview
This module has a Web user interface to use GridScanner. It employs two other modules ( `CUTIE Model Server` and `Invoice Placeholder API`) to process an invoice. The user will submit an invoice using Browser, and GridScanner will scan for invoice details. After invoice processing, it will produce an Excel file that can be downloaded or uploaded to the user's drive after Google authentication.

## Tech Stack
* `Ptyhon`: It is easier to write programs in Python compared to other Scripting Languages. It is also enriched with a lot of useful libraries.

* `FastAPI`: It is a modern Python web framework designed to: provide a lightweight microframework with an intuitive, Flask-like routing system. 
* `ASGI server`: ASGI server made FastAPI easy to run an async event loop that counts incoming requests.

## How to run
* Run `pip install -r requrements.txt` to install required python libraries.
* Run `uvicorn app:app` to start the server.
* Go to http://127.0.0.1:8000 to access the UI.

## Challenges faced
* Handled asyncronous call to modules.
* Request and response of multipart/form-data.