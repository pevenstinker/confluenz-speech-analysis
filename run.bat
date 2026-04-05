@echo off
cd /d "%~dp0"

if "%1"=="label" (
    shift
    .venv\Scripts\python.exe label_file.py %*
) else if "%1"=="classify" (
    shift
    .venv\Scripts\python.exe classify_file.py %*
) else if "%1"=="train" (
    shift
    .venv\Scripts\python.exe train.py %*
) else (
    .venv\Scripts\python.exe main.py %*
)
