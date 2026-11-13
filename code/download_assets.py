"""Download official CIFAR-10 and optional pretrained ResNet-18 weights.

Existing files are hashed before use; network TLS verification stays enabled.
"""
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EE6483_DATA_DIR', ROOT / 'datasets'))
CIFAR_MD5 = 'c58f30108f718f92721af3b95e74349a'
RESNET_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, destination, *, expected_md5=None, expected_sha256=None):
    def verified(path):
        return ((expected_md5 is None or md5(path) == expected_md5)
                and (expected_sha256 is None or sha256(path) == expected_sha256))

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and verified(destination):
        print(f'[verified] {destination}')
        return
    part = destination.with_name(destination.name + '.part')
    try:
        with urllib.request.urlopen(url, timeout=90) as response, open(part, 'wb') as f:
            while block := response.read(1024 * 1024):
                f.write(block)
        if not verified(part):
            raise ValueError(f'Hash mismatch for {destination}: '
                             f'MD5={md5(part)} SHA256={sha256(part)}')
        part.replace(destination)
    finally:
        if part.exists():
            part.unlink()
    print(f'[verified] {destination}')


def extract_cifar(archive, destination):
    marker = destination / 'cifar-10-batches-py' / 'test_batch'
    if marker.is_file():
        return
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe tar member: {member.name}')
        tar.extractall(destination, members=members)
    print(f'[extracted] {destination}')


if __name__ == '__main__':
    cifar_root = DATA / 'cifar10'
    archive = cifar_root / 'cifar-10-python.tar.gz'
    fetch('https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz',
          archive, expected_md5=CIFAR_MD5)
    extract_cifar(archive, cifar_root)
    # Only needed if training the Dogs-vs-Cats ResNet from ImageNet weights.
    if os.environ.get('EE6483_DOWNLOAD_RESNET') == '1':
        hub = Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
        fetch('https://download.pytorch.org/models/resnet18-f37072fd.pth',
              hub, expected_sha256=RESNET_SHA256)
