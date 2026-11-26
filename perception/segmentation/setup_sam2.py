"""Fetch the pinned, unmodified official source and verified model weights."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
SHA256 = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = ROOT/'third_party'/'sam2'
    marker = ROOT/'third_party'/'sam2_commit.txt'
    if source.exists():
        if not marker.exists() or marker.read_text().strip() != COMMIT:
            raise RuntimeError('Existing source has a different or missing commit marker; preserve it and resolve manually.')
    else:
        response = requests.get(f'https://codeload.github.com/facebookresearch/sam2/zip/{COMMIT}', timeout=180)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for member in archive.infolist():
                relative = Path(*Path(member.filename).parts[1:])
                target = (source/relative).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError('Unsafe archive path')
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
        marker.write_text(COMMIT+'\n', encoding='utf-8')
    weights = ROOT/'models'/'sam2.1_hiera_tiny.pt'
    weights.parent.mkdir(exist_ok=True)
    if not weights.exists() or digest(weights) != SHA256:
        temporary = weights.with_suffix('.part')
        with requests.get(URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk)
        if digest(temporary) != SHA256:
            raise RuntimeError('Checkpoint SHA256 mismatch')
        temporary.replace(weights)
    (weights.parent/'sam2_source.json').write_text(json.dumps(dict(
        repo='https://github.com/facebookresearch/sam2', commit=COMMIT,
        checkpoint_url=URL, checkpoint_bytes=weights.stat().st_size, sha256=SHA256
    ), indent=2), encoding='utf-8')
    print('SAM 2 source and checkpoint ready.')


if __name__ == '__main__':
    main()
