"""Resolve the DANDI 000044 Achilles session NWB asset to a streamable S3 URL."""
import json
import urllib.request
import requests

DANDISET_ID = "000044"
ASSET_SUBSTR = "ses-Achilles-1025"


def api_get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def find_asset(dandiset_id, substr):
    page_url = f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/assets/"
    while page_url:
        data = api_get(page_url)
        for a in data["results"]:
            if substr in a["path"] and a["path"].endswith(".nwb"):
                # skip the 2024-11 remade derivative if present; prefer the classic file
                return a
        page_url = data.get("next")
    raise RuntimeError("asset not found")


asset = find_asset(DANDISET_ID, ASSET_SUBSTR)
print("ASSET PATH:", asset["path"])
print("ASSET ID:", asset["asset_id"])
download_url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/assets/{asset['asset_id']}/download/"
r = requests.get(download_url, allow_redirects=False, timeout=60)
if r.status_code != 302:
    print("STATUS:", r.status_code, r.text[:500])
r.raise_for_status()
s3_url = r.headers["Location"]
print("S3 URL HEAD:", s3_url[:120], "...")

with open("_asset_info.json", "w") as f:
    json.dump({"path": asset["path"], "asset_id": asset["asset_id"], "s3_url": s3_url}, f, indent=2)
print("saved _asset_info.json")