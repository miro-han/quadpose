@echo off
setlocal
cd /d "%~dp0..\.."
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" -m perception.segmentation.segment_ui
if not errorlevel 1 exit /b 0
echo Tracking UI failed. See the error above.
pause
exit /b 1
:missing
echo Set up the project Python environment as described in README.txt first.
pause
exit /b 1
