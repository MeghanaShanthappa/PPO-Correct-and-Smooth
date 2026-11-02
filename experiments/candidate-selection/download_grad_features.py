"""Retrieve only the two required Products members of the author-linked ZIP.

HTTP ranges preserve ZIP member boundaries; downloads are resumable. Validate
uncompressed length, ZIP CRC32 and NumPy shape, and record SHA256 provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
import requests

URL = 'https://archive.org/download/gradbert_embeddings/embeddings.zip'
MEMBERS = ('giant_feats.npy', 'gradbert_kd_feats.npy')


def get_range(start, end, key):
    response = requests.get(URL + '?member=' + key, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                            stream=True, timeout=(30, 120))
    response.raise_for_status()
    expected = f'bytes {start}-{end}/31049240898'
    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
        response.close()
        raise RuntimeError('server did not honor exact archive range')
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.index).read_text())
    manifests = []
    for name in MEMBERS:
        info = next(f for f in index if f['name'] == 'embeddings/ogbn-products/' + name)
        destination = out / name
        record = out / (name + '.json')
        if destination.exists() and record.exists():
            old = json.loads(record.read_text())
            if hashlib.sha256(destination.read_bytes()).hexdigest() != old['sha256']:
                raise RuntimeError('existing completed feature changed')
            manifests.append(old)
            continue
        with get_range(info['offset'], info['offset'] + 29, name + '-header') as response:
            header = response.content
        assert header[:4] == b'PK\x03\x04'
        method = struct.unpack_from('<H', header, 8)[0]
        n, extra = struct.unpack_from('<HH', header, 26)
        assert method == 8
        start = info['offset'] + 30 + n + extra
        compressed = out / (name + '.deflate.part')
        done = compressed.stat().st_size if compressed.exists() else 0
        total = info['compressed_bytes']
        if done > total:
            raise RuntimeError('partial download larger than member')
        for attempt in range(5):
            try:
                if done < total:
                    print(f'Downloading {name}: resume {done}/{total}', flush=True)
                    with get_range(start + done, start + total - 1, name + '-data') as response, compressed.open('ab') as handle:
                        last = done // (256 * 1024**2)
                        for block in response.iter_content(8 * 1024**2):
                            if not block:
                                continue
                            handle.write(block)
                            done += len(block)
                            if done // (256 * 1024**2) > last:
                                last = done // (256 * 1024**2)
                                print(f'{name}: {done}/{total} compressed bytes', flush=True)
                if done != total:
                    raise RuntimeError('incomplete compressed member')
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                done = compressed.stat().st_size if compressed.exists() else 0
                time.sleep(2)
        temporary = out / (name + '.extracting')
        decoder, crc, digest, count = zlib.decompressobj(-15), 0, hashlib.sha256(), 0
        with compressed.open('rb') as source, temporary.open('wb') as target:
            while block := source.read(8 * 1024**2):
                value = decoder.decompress(block)
                target.write(value)
                digest.update(value)
                crc = zlib.crc32(value, crc)
                count += len(value)
            value = decoder.flush()
            target.write(value)
            digest.update(value)
            crc = zlib.crc32(value, crc)
            count += len(value)
        if not decoder.eof or count != info['bytes'] or crc != info['crc32']:
            raise RuntimeError('ZIP integrity check failed')
        feature = np.load(temporary, mmap_mode='r', allow_pickle=False)
        if feature.shape != (2449029, 768) or feature.dtype != np.float32:
            raise RuntimeError('unexpected Products embedding shape or dtype')
        manifest = {'source': URL, 'author_mirror_reference': 'https://github.com/cmavro/GRAD/issues/1#issuecomment-1553526371',
                    'member': info['name'], 'bytes': count, 'crc32': crc, 'sha256': digest.hexdigest(),
                    'shape': list(feature.shape), 'dtype': str(feature.dtype)}
        del feature
        temporary.replace(destination)
        record.write_text(json.dumps(manifest, indent=2) + '\n')
        manifests.append(manifest)
        print(f'Verified {name}: {digest.hexdigest()}', flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')


if __name__ == '__main__':
    main()
