#!/usr/bin/env python3
"""Download Numerai v5.0 dataset files via the public GraphQL API (no numerapi
dependency; this replicates numerapi's dataset() call: the API returns a signed
S3 URL per filename). Files land in <run-dir>/data/.

Usage: python3 download_numerai.py <dest_dir> <filename> [<filename> ...]
"""
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api-tournament.numer.ai/"
UA = "numerapi/2.19.1"


def gql(query, variables=None):
    q = {"query": query, "variables": variables or {}}
    req = urllib.request.Request(
        API, data=json.dumps(q).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def signed_url(filename):
    data = gql("query($filename: String!) { dataset(filename: $filename) }",
               {"filename": filename})
    return data["data"]["dataset"]


def download(filename, dest_dir):
    dest = Path(dest_dir) / filename.replace("/", "_")
    url = signed_url(filename)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        total = int(r.headers.get("Content-Length", 0))
        print(f"{filename}: {total/1e9:.2f} GB -> {dest}", flush=True)
        done = 0
        next_report = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"  {filename}: {done/1e9:.2f}/{total/1e9:.2f} GB",
                          flush=True)
                    next_report += 500_000_000
    print(f"DONE {filename}: {done} bytes", flush=True)


if __name__ == "__main__":
    dest_dir = sys.argv[1]
    for fn in sys.argv[2:]:
        download(fn, dest_dir)
