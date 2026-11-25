#!/bin/bash
echo "Installing dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Starting the application..."
gunicorn -w 5 -k uvicorn.workers.UvicornWorker main:app