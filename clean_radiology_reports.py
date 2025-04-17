import os
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
import pandas as pd

ARCHIVE_DIR = "archive"
IMAGES_DIR = os.path.join(ARCHIVE_DIR, "images")
REPORTS_DIR = os.path.join(ARCHIVE_DIR, "reports")

IMAGE_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_dcm.tgz"
REPORT_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"

IMAGE_TGZ = os.path.join(ARCHIVE_DIR, "NLMCXR_dcm.tgz")
REPORT_TGZ = os.path.join(ARCHIVE_DIR, "NLMCXR_reports.tgz")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def download_and_extract(url, tgz_path, extract_to):
    if not os.path.exists(tgz_path):
        print(f"⬇️ Downloading {url}")
        urllib.request.urlretrieve(url, tgz_path)
    else:
        print(f"✅ Already downloaded: {tgz_path}")

    print(f"📦 Extracting {tgz_path} to {extract_to}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

download_and_extract(IMAGE_URL, IMAGE_TGZ, IMAGES_DIR)
download_and_extract(REPORT_URL, REPORT_TGZ, REPORTS_DIR)

XML_FOLDER = os.path.join(REPORTS_DIR, "NLMCXR_reports/ecgen-radiology")
report_rows = []
projection_rows = []

for filename in os.listdir(XML_FOLDER):
    if not filename.endswith(".xml"):
        continue

    path = os.path.join(XML_FOLDER, filename)
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        uid = root.findtext("uId")
        if not uid:
            uid = os.path.splitext(filename)[0]

        mesh = ""
        problems = ""
        image = ""
        indication = ""
        comparison = ""
        findings = ""
        impression = ""

        mesh_terms = root.findall("MeSH/major")
        mesh = "; ".join([m.text for m in mesh_terms if m.text]) if mesh_terms else ""

        problem_el = root.find("PROBLEM")
        problems = problem_el.text if problem_el is not None else ""

        for abstract in root.iter("AbstractText"):
            label = abstract.attrib.get("Label", "").upper()
            if label == "INDICATION":
                indication = abstract.text or ""
            elif label == "COMPARISON":
                comparison = abstract.text or ""
            elif label == "FINDINGS":
                findings = abstract.text or ""
            elif label == "IMPRESSION":
                impression = abstract.text or ""

        parent = root.find("parentImage")
        if parent is not None:
            image_id = parent.attrib.get("id")
            if image_id:
                image = image_id + ".jpg"

        report_rows.append({
            "uid": uid,
            "MeSH": mesh,
            "Problems": problems,
            "image": image,
            "indication": indication,
            "comparison": comparison,
            "findings": findings,
            "impression": impression
        })

        for parent_image in root.findall("parentImage"):
            projection_id = parent_image.attrib.get("id", "")
            filename_img = projection_id + ".jpg"
            view = "Lateral" if "4001" in projection_id else "PA"
            projection_rows.append({
                "uid": uid,
                "projection_id": projection_id,
                "view": view,
                "filename": filename_img
            })

    except Exception as e:
        print(f"❌ Failed to process {filename}: {e}")

df_reports = pd.DataFrame(report_rows)
df_projections = pd.DataFrame(projection_rows)

df_reports.to_csv(os.path.join(ARCHIVE_DIR, "indiana_reports.csv"), index=False)
df_projections.to_csv(os.path.join(ARCHIVE_DIR, "indiana_projections.csv"), index=False)

print("✅ CSVs created in archive/: indiana_reports.csv and indiana_projections.csv")
