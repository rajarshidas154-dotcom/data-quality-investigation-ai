@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe python -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m dqa.demo
if errorlevel 1 exit /b 1
echo Setup complete. Run start.cmd and open http://127.0.0.1:8765
