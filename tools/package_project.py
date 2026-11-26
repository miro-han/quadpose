"""Build a small handoff ZIP without runtime environments or private recordings."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {'.venv', '.camera_env', '__pycache__', '.git', 'archive'}


def main():
    paths = [ROOT/'README.md', ROOT/'pyproject.toml', ROOT/'.gitignore', ROOT/'.gitattributes'] + list(ROOT.glob('*.cmd'))
    for folder in ['capture', 'rig', 'perception', 'pose', 'evaluation', 'experiments', 'docs', 'tools']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and not SKIP.intersection(p.relative_to(ROOT).parts)
                     and p.suffix not in {'.pyc', '.log'})
    paths += [ROOT/'models/sam2_source.json', ROOT/'third_party/sam2_commit.txt']
    evidence = {
        'outputs/camera/recorder_checks/hardware_report.json': 'verification/hardware_report.json',
        'outputs/segmentation/20261124_215508_e1cb28/session.json': 'verification/sam2_session.json',
        'outputs/segmentation/20261124_215508_e1cb28/frames.jsonl': 'verification/sam2_frames.jsonl',
        'outputs/segmentation/20261124_215508_e1cb28/review.png': 'verification/sam2_review.png',
        'outputs/segmentation/20261124_215749_db9b73/session.json': 'verification/gui_session.json',
    }
    manifest = []
    output = ROOT/'dist/Team7_Project_Handoff.zip'
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        entries = [(p, p.relative_to(ROOT).as_posix()) for p in sorted(set(paths))]
        entries += [(ROOT/source, target) for source, target in evidence.items()]
        for path, relative in entries:
            if not path.is_file():
                continue
            data = path.read_bytes()
            archive.writestr('Team7_Plate_Tracking/'+relative, data)
            manifest.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr('Team7_Plate_Tracking/verification/README.txt',
                        'Evidence copied from the original local tests. Absolute source/session paths are historical; '
                        'videos and complete outputs are not bundled. sam2_session: 3-frame CPU test; '
                        'gui_session: 2-frame UI test; hardware_report: 2 physical cameras.\n')
        archive.writestr('Team7_Plate_Tracking/package_manifest.json',
                        json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert all(not SKIP.intersection(Path(name).parts) for name in archive.namelist())
    temporary.replace(output)
    print(f'{output}: {output.stat().st_size / 1024**2:.2f} MiB, {len(manifest)} files + manifest/readme')


if __name__ == '__main__':
    main()
