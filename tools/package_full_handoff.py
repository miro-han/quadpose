"""Package project, official model/source, results and the two project videos."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {'.venv', '.camera_env', '__pycache__', '.git', 'node_modules', 'archive'}
VIDEO_NAMES = ['WIN_20261124_20_05_34_Pro.mp4', 'WIN_20261124_20_04_41_Pro.mp4']


def main():
    entries = [(p, p.relative_to(ROOT).as_posix()) for p in
               [ROOT/'README.md', ROOT/'pyproject.toml', ROOT/'.gitignore', ROOT/'.gitattributes', *ROOT.glob('*.cmd')]]
    entries.append((ROOT/'docs/FULL_PACKAGE_START_HERE.txt', '00_README_FIRST.txt'))
    for directory in ['capture', 'rig', 'perception', 'pose', 'evaluation', 'experiments', 'docs', 'models',
                      'third_party', 'outputs', 'recordings', 'tools']:
        for p in (ROOT/directory).rglob('*'):
            relative = p.relative_to(ROOT)
            if p.is_file() and not SKIP.intersection(relative.parts) and p.suffix not in {'.pyc', '.log', '.tmp', '.part'}:
                entries.append((p, relative.as_posix()))
    for name in VIDEO_NAMES:
        p = ROOT/'data'/'sample_videos'/name
        if not p.exists():
            p = Path.home()/'Pictures'/'Camera Roll'/name
        if not p.is_file():
            raise FileNotFoundError(p)
        entries.append((p, 'data/sample_videos/'+name))
    output = ROOT.parent/'Team7_Complete_Handoff.zip'
    temp = output.with_suffix('.zip.tmp')
    manifest = []
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for p, relative in sorted(entries, key=lambda item: item[1]):
            before = p.stat()
            digest = hashlib.sha256()
            with p.open('rb') as source, archive.open('Team7_Plate_Tracking/'+relative, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(1024*1024), b''):
                    digest.update(block)
                    target.write(block)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'File changed while packaging: {p}')
            manifest.append(dict(path=relative, bytes=after.st_size, sha256=digest.hexdigest()))
        archive.writestr('Team7_Plate_Tracking/FULL_PACKAGE_MANIFEST.json',
                        json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        for entry in manifest:
            digest = hashlib.sha256()
            with archive.open('Team7_Plate_Tracking/'+entry['path']) as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    digest.update(block)
            assert digest.hexdigest() == entry['sha256'], entry['path']
    temp.replace(output)
    print(json.dumps(dict(zip=str(output), bytes=output.stat().st_size,
                         files=len(manifest), integrity='CRC and all SHA256 verified'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
