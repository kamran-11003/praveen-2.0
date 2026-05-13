# download_avatar.py
import urllib.request
import os

url = "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@1.7/avatars/brunette.glb"
out = os.path.join(os.path.dirname(__file__), "brunette.glb")

print(f"Downloading {url} ...")
urllib.request.urlretrieve(url, out)
print(f"Saved to: {out}")