"""Resolve the streaming URL for the Achilles session in DANDI 000044."""
import requests

DANDISET_ID = "000044"
TARGET = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"

# List assets via the DANDI API (versioned endpoint)
r = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/draft/assets/",
    params={"page_size": 100},
    timeout=30,
)
r.raise_for_status()
data = r.json()
assets = data.get("results", data) if isinstance(data, dict) else data
assets = [a for a in assets if a["path"] == TARGET]
assert len(assets) >= 1, f"no asset found for {TARGET}"

assert len(assets) >= 1, f"no asset found for {TARGET}"
asset = assets[0]
asset_id = asset["asset_id"]
print("asset_id:", asset_id)
print("path:", asset["path"])
print("size:", asset.get("size"))

# Resolve the s3_url: the /download/ endpoint redirects (302) to a GET-only
# presigned URL. HEAD on the redirect target 403s, so grab the Location header
# from a GET with allow_redirects=False.
r = requests.get(
    f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
    allow_redirects=False,
    timeout=30,
)
r.raise_for_status()
s3_url = r.headers["Location"]
print("s3_url:", s3_url)

with open("s3_url.txt", "w") as f:
    f.write(s3_url)
print("saved to s3_url.txt")