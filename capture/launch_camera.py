"""Portable launcher: reuse only a working Python environment; log UI errors."""
import subprocess
import sys
import traceback
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = 'import cv2, numpy, PIL.ImageTk, cv2_enumerate_cameras, tkinter'


def main():
    candidates = [HERE.parent / '.venv/Scripts/python.exe',
                  HERE / '.camera_env/Scripts/python.exe', Path(sys.executable)]
    python = None
    with (HERE / 'launcher.log').open('w', encoding='utf-8') as log:
        for candidate in candidates:
            if candidate.exists() and subprocess.run([str(candidate), '-c', CHECK], stdout=log, stderr=log).returncode == 0:
                python = candidate
                break
        if python is None:
            print('Preparing dependencies. Please keep this window open.', flush=True)
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(HERE / '.camera_env')
            python = HERE / '.camera_env/Scripts/python.exe'
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(HERE / 'requirements.txt')], check=True)
            subprocess.run([str(python), '-c', CHECK], check=True)
        print(f'Using: {python}', flush=True)
        args = ['--help'] if '--check' in sys.argv else ['--preview']
        result = subprocess.run([str(python), '-m', 'capture.camera_recorder', *args], cwd=HERE.parent, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError('Camera UI failed; see capture/launcher.log')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
