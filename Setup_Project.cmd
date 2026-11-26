@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto python
py -3 -m venv .venv
goto created
:python
python -m venv .venv
:created
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10 or newer required'"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -e . -r capture\requirements.txt -r perception\segmentation\requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m perception.segmentation.setup_sam2
if errorlevel 1 goto failed
echo Setup complete. Open 01_Camera_Recorder.cmd or 02_Object_Tracking.cmd.
pause
exit /b 0
:failed
echo Setup failed. Check the error above. Python 3.10 or newer and internet are required.
pause
exit /b 1
