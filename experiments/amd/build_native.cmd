@echo off
setlocal
cd /d "%~dp0"
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%v in (`"%vswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vsroot=%%v"
if not defined vsroot exit /b 1
call "%vsroot%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /LD /I".venv\ort-sdk" /Fo".venv\explicit_dml.obj" explicit_dml.cpp /link dxgi.lib d3d12.lib /OUT:".venv\explicit_dml.dll" /IMPLIB:".venv\explicit_dml.lib"
exit /b %errorlevel%
