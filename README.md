# OrgDNA MVP — Enterprise Memory Platform demo

A Streamlit app that answers questions from ~10 Northeastern ITS documents.
This is the MVP demo for the OrgDNA class project.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free (so the class can scan a QR code)
1. Push this `app/` folder to a public GitHub repo (e.g. `orgdna-mvp`).
2. Go to https://share.streamlit.io → "Create app" → pick the repo,
   branch `main`, main file `app.py`.
3. (Optional, for AI-generated answers) In the app's Settings → Secrets, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Without a key the app still works in extractive-search mode.
4. Your app gets a public URL like `https://orgdna-mvp.streamlit.app`.
5. Generate the QR code for that URL:
   ```bash
   pip install qrcode[pil]
   python make_qr.py https://orgdna-mvp.streamlit.app
   ```
   Drop `orgdna_qr.png` onto the last slide of the deck.
