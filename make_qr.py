"""Generate a QR code PNG for the deployed app URL.
Usage: python make_qr.py https://your-app.streamlit.app
"""
import sys
import qrcode

url = sys.argv[1] if len(sys.argv) > 1 else "https://orgdna-mvp.streamlit.app"
img = qrcode.make(url, box_size=12, border=2)
img.save("orgdna_qr.png")
print(f"Saved orgdna_qr.png -> {url}")
