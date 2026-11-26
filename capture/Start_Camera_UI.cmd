@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 launch_camera.py %*
goto finished
:use_python
python launch_camera.py %*
:finished
if not errorlevel 1 exit /b 0
echo Could not start. Check the error above and capture\launcher.log.
pause
exit /b 1
