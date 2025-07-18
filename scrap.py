import os
import re
import requests
from bs4 import BeautifulSoup

# Set the base listing URL for astro-ph, 2025
BASE_URL = "https://arxiv.org/list/astro-ph/25?show=1000"
PDF_BASE = "https://arxiv.org/pdf"

# Folder to save PDFs
SAVE_DIR = "arxiv_pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fetch listing page
res = requests.get(BASE_URL)
if res.status_code != 200:
    print("Failed to fetch the page.")
    exit()

soup = BeautifulSoup(res.text, 'html.parser')

# Extract all arXiv IDs using regex
id_pattern = re.compile(r'arXiv:(\d+\.\d+)')
ids = id_pattern.findall(soup.get_text())

# Remove duplicates
ids = list(set(ids))
print(f"Found {len(ids)} unique paper IDs.")

# Download PDFs
for arxiv_id in ids:
    pdf_url = f"{PDF_BASE}/{arxiv_id}.pdf"
    filename = os.path.join(SAVE_DIR, f"{arxiv_id}.pdf")

    if os.path.exists(filename):
        print(f"[✓] Already exists: {arxiv_id}")
        continue

    print(f"[↓] Downloading: {arxiv_id}")
    pdf_res = requests.get(pdf_url)
    if pdf_res.status_code == 200:
        with open(filename, "wb") as f:
            f.write(pdf_res.content)
    else:
        print(f"[✗] Failed to download: {arxiv_id} (Status {pdf_res.status_code})")

print("✅ All available PDFs downloaded.")
