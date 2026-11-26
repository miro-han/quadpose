param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$Project = Split-Path $PSScriptRoot -Parent
$Venv = Join-Path $Project '.venv'
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
$Lock = Join-Path $PSScriptRoot 'requirements-win-py312.lock.txt'
$CheckPython = "import struct, sys; sys.exit(0 if sys.platform == 'win32' and sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 'This setup requires Windows x64 and Python 3.12.')"

function Invoke-Python([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable exited with code $LASTEXITCODE."
    }
}

if (!(Test-Path -LiteralPath $VenvPython)) {
    Invoke-Python $Python @('-c', $CheckPython)
    Invoke-Python $Python @('-m', 'venv', $Venv)
}
Invoke-Python $VenvPython @('-c', $CheckPython)
$VenvConfig = Join-Path $Venv 'pyvenv.cfg'
if (!(Select-String -LiteralPath $VenvConfig -Pattern '^include-system-site-packages\s*=\s*false\s*$' -Quiet)) {
    throw 'The existing .venv includes system site-packages. Create an isolated .venv before running this setup.'
}

Invoke-Python $VenvPython @('-m', 'pip', 'install', '-r', $Lock)
Invoke-Python $VenvPython @('-m', 'pip', 'install', '--no-cache-dir', '--no-deps', '--index-url', 'https://download.pytorch.org/whl/cu126', 'torch==2.8.0+cu126', 'torchvision==0.23.0+cu126')
Invoke-Python $VenvPython @('-m', 'pip', 'check')
$CheckCuda = "import sys, torch; sys.exit('CUDA unavailable. Check the NVIDIA driver and PyTorch build.') if not torch.cuda.is_available() else print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.ones(1, device='cuda').item())"
Invoke-Python $VenvPython @('-c', $CheckCuda)
