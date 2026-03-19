import os, xnat
from dotenv import load_dotenv
load_dotenv()
with xnat.connect(os.getenv("XNAT_URL"), user=os.getenv("XNAT_USER"), password=os.getenv("XNAT_PASS")) as s:
    proj = s.projects["ABIDE"]
    subjects = list(proj.subjects.values())
    print("Total subjects in ABIDE project:", len(subjects))
    print("Example labels:", [subj.label for subj in subjects[:5]])
