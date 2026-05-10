@echo off
cd /d "D:\ml chat bot\ml-chatbot\backend"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
