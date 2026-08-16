# Catalog all assets in 001262, save list for sampling
import requests
import pandas as pd

DANDI = "001262"
VER = "0.241205.0959"
url = f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}/assets/"
rows = []
page = 1
while url:
    r = requests.get(url, params={"page_size": 100} if page == 1 else None)
    d = r.json()
    for a in d["results"]:
        rows.append({"path": a["path"], "asset_id": a["asset_id"], "size": a["size"]})
    url = d["next"]
    page += 1
    print(f"page {page}, total {len(rows)}")

df = pd.DataFrame(rows)
df["subject"] = df["path"].str.split("/").str[0]
df["is_optical"] = df["path"].str.contains("nm")
df.to_csv("asset_catalog.csv", index=False)
print(df.shape)
print(df.groupby("subject").size())
print("non-optical:", (~df["is_optical"]).sum())
