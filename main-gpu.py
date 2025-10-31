import os
import cv2
import re
import subprocess
from PIL import Image
import imagehash
from paddleocr import PaddleOCR
import pandas as pd
from rapidfuzz import fuzz
from email_validator import validate_email, EmailNotValidError
import json
from datetime import datetime


VIDEO_PATH = "Screencast from 2025-10-16 15-16-43.webm"
FRAMES_DIR = "frames"
FRAME_FPS = 10
EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHASH_THRESHOLD = 3
DEDUP_RATIO = 96
LOCAL_PART_RATIO = 90
DOMAIN_PART_RATIO = 70

HIGH_CONFIDENCE_REPLACEMENTS = {
    "qmail.com": "gmail.com",
    "hotmall.com": "hotmail.com",
    "yaho.com": "yahoo.com",
    "gmai.com": "gmail.com",
}

KNOWN_DOMAINS = [
    "gmail.com",
    "hotmail.com",
    "yahoo.com",
    "outlook.com",
    "aol.com",
    "icloud.com",
    "protonmail.com",
    "zoho.com",
    "live.com",
    "msn.com",
    "gmx.com",
    "mail.com",
]

DOMAIN_CORRECTION_THRESHOLD = 90


def extract_frames(video_path, frames_dir, fps):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            os.path.join(frames_dir, "frame_%04d.png"),
        ],
        check=True,
    )


def unique_frames(frames_dir):
    hashes = []
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    unique = []
    for fname in frames:
        path = os.path.join(frames_dir, fname)
        img = Image.open(path)
        phash = imagehash.phash(img)
        if all(phash - h > PHASH_THRESHOLD for h in hashes):
            hashes.append(phash)
            unique.append(fname)
        # unique.append(fname)
    return unique


def crop_roi(image, roi=None):
    if roi is None:
        return image
    y0, y1, x0, x1 = roi
    return image[y0:y1, x0:x1]


def extract_emails_from_frames(frames_dir, unique_frames, roi=None):
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=True)
    # normalizer = ImageNormalizer()
    results = []

    for idx, fname in enumerate(unique_frames):
        path = os.path.join(frames_dir, fname)
        img_cv = cv2.imread(path)
        if roi:
            img_cv = crop_roi(img_cv, roi)
        # image = normalizer.normalize_image(img_cv)
        # img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        ocr_res = ocr.ocr(path, cls=True)
        frame_text = " ".join([line[1][0] for line in ocr_res[0] if line[1][1] > 0.5])

        for email in re.findall(EMAIL_REGEX, frame_text):
            try:
                # _ = validate_email(email)
                conf = [line[1][1] for line in ocr_res[0] if email in line[1][0]]
                results.append(
                    {
                        "email": email,
                        "frame": fname,
                        "conf": sum(conf) / len(conf) if conf else 0.5,
                        "sec": idx / FRAME_FPS,
                    }
                )
            except EmailNotValidError:
                continue
    return results


def normalize_domain(full_domain):
    domain_lower = full_domain.lower()

    if domain_lower in HIGH_CONFIDENCE_REPLACEMENTS:
        return HIGH_CONFIDENCE_REPLACEMENTS[domain_lower]

    best_match_domain = full_domain
    best_score = 0

    for known in KNOWN_DOMAINS:
        score = fuzz.ratio(domain_lower, known.lower())

        if score > best_score:
            best_score = score
            best_match_domain = known

    if best_score >= DOMAIN_CORRECTION_THRESHOLD:
        return best_match_domain
    else:
        return full_domain


def deduplicate_emails(raw_results):
    emails = {}

    def split_email(email):
        parts = email.lower().split("@")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    for entry in raw_results:
        try:
            local_part, full_domain = entry["email"].rsplit("@", 1)
        except ValueError:
            continue
        corrected_domain = normalize_domain(full_domain)
        email = f"{local_part}@{corrected_domain}"

        local, domain = split_email(email)

        if not local or not domain:
            continue

        found = False

        for existing in emails:
            existing_local, existing_domain = split_email(existing)

            local_match = fuzz.ratio(local, existing_local) >= LOCAL_PART_RATIO
            domain_match = fuzz.ratio(domain, existing_domain) >= DOMAIN_PART_RATIO

            if local_match and domain_match:
                group = emails[existing]
                group["frames_seen"].add(entry["frame"])
                group["confidences"].append(entry["conf"])
                group["last_seen_sec"] = max(group["last_seen_sec"], entry["sec"])
                found = True
                break

        if not found:
            emails[email] = {
                "frames_seen": set([entry["frame"]]),
                "confidences": [entry["conf"]],
                "first_seen_sec": entry["sec"],
                "last_seen_sec": entry["sec"],
                "sample_frame": entry["frame"],
            }

    final = []
    for email, data in emails.items():
        mean_conf = (
            sum(data["confidences"]) / len(data["confidences"])
            if data["confidences"]
            else 0.0
        )
        final.append(
            {
                "email": email,
                "frames_seen": len(data["frames_seen"]),
                "mean_conf": mean_conf,
                "first_seen_sec": data["first_seen_sec"],
                "last_seen_sec": data["last_seen_sec"],
                "sample_frame": data["sample_frame"],
            }
        )
    return final


def save_to_csv(records, filename="output.csv"):
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)


def save_to_json(records, filename="output.json"):
    df = pd.DataFrame(records)
    df.to_json(filename, orient="records", force_ascii=False, indent=4)


def cleanup_frames(frames_dir):
    for fname in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, fname))
    os.rmdir(frames_dir)


if __name__ == "__main__":
    print(datetime.now())

    print("Extracting frames...")
    extract_frames(VIDEO_PATH, FRAMES_DIR, FRAME_FPS)

    print("Selecting unique frames...")
    unique = unique_frames(FRAMES_DIR)

    print(f"{len(unique)} unique frames found.")
    with open("unique-frames.json", "w", encoding="utf-8") as f:
        json_str = json.dumps(unique, ensure_ascii=False, default=str)
        f.write(json_str)

    print("Running OCR & extracting emails...")
    results = extract_emails_from_frames(FRAMES_DIR, unique)
    with open("raw-results-frames.json", "w", encoding="utf-8") as f:
        json_str = json.dumps(results, ensure_ascii=False, default=str)
        f.write(json_str)

    print(f"Found {len(results)} candidate emails, deduplicating...")
    deduped = deduplicate_emails(results)

    print(f"{len(deduped)} unique validated emails exported.")
    save_to_csv(deduped)
    save_to_json(deduped)

    print("Cleaning up temporary frames...")
    cleanup_frames(FRAMES_DIR)

    print("Done. Output saved to output.csv.")

    print(datetime.now())
