import os
import cv2
import re
import subprocess
import shutil
import datetime
from PIL import Image
import imagehash
from paddleocr import PaddleOCR
import pandas as pd
from rapidfuzz import fuzz
from email_validator import validate_email, EmailNotValidError
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import lru_cache
import json


class VideoProcessor:
    def __init__(
        self,
        video_path,
        frames_dir="frames",
        frame_fps=5,
        use_gpu=True,
        roi=None,
        keep_frames=False,
        phash_threshold=3,
        local_part_ratio=90,
        domain_part_ratio=70,
        domain_correction_threshold=90,
        high_confidence_replacements=None,
        known_domains=None,
    ):
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.frame_fps = frame_fps
        self.use_gpu = use_gpu
        self.roi = roi
        self.keep_frames = keep_frames
        self.phash_threshold = phash_threshold
        self.local_part_ratio = local_part_ratio
        self.domain_part_ratio = domain_part_ratio
        self.domain_correction_threshold = domain_correction_threshold

        self.email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        self.high_confidence_replacements = high_confidence_replacements or {
            "qmail.com": "gmail.com",
            "hotmall.com": "hotmail.com",
            "yaho.com": "yahoo.com",
            "gmai.com": "gmail.com",
        }
        self.known_domains = known_domains or [
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

    # ---------------- FRAME EXTRACTION ----------------

    def _extract_frames(self):
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                self.video_path,
                "-vf",
                f"fps={self.frame_fps}",
                os.path.join(self.frames_dir, "frame_%04d.png"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ---------------- FRAME HASHING ----------------

    def _compute_phash(self, path):
        with Image.open(path) as img:
            return imagehash.phash(img)

    def _unique_frames(self, return_all=False):
        frames = sorted([f for f in os.listdir(self.frames_dir) if f.endswith(".png")])
        if not frames:
            return []
        
        if return_all:
            return frames

        with ThreadPoolExecutor() as ex:
            hashes = list(
                tqdm(
                    ex.map(
                        lambda f: self._compute_phash(os.path.join(self.frames_dir, f)),
                        frames,
                    ),
                    total=len(frames),
                    desc="Hashing frames",
                )
            )

        unique = [frames[0]]
        unique_hashes = [hashes[0]]

        for fname, h in zip(frames[1:], hashes[1:]):
            if all(abs(h - uh) > self.phash_threshold for uh in unique_hashes):
                unique.append(fname)
                unique_hashes.append(h)
        return unique

    # ---------------- OCR ----------------

    def _crop_roi(self, image):
        if self.roi is None:
            return image
        y0, y1, x0, x1 = self.roi
        return image[y0:y1, x0:x1]

    def _extract_emails_from_frames(self, unique_frames):
        ocr = PaddleOCR(
            use_angle_cls=True, lang="en", use_gpu=self.use_gpu, show_log=False
        )
        results = []

        for idx, fname in enumerate(tqdm(unique_frames, desc="Running OCR")):
            path = os.path.join(self.frames_dir, fname)
            img_cv = cv2.imread(path)
            if self.roi:
                img_cv = self._crop_roi(img_cv)

            ocr_res = ocr.ocr(img_cv, cls=True)
            if not ocr_res or not ocr_res[0]:
                continue

            frame_text = " ".join(
                [line[1][0] for line in ocr_res[0] if line[1][1] > 0.5]
            )

            for email in re.findall(self.email_regex, frame_text):
                try:
                    conf = [line[1][1] for line in ocr_res[0] if email in line[1][0]]
                    # validate_email(email)
                    results.append(
                        {
                            "email": email,
                            "frame": fname,
                            "conf": sum(conf) / len(conf) if conf else 0.5,
                            "sec": idx / self.frame_fps,
                        }
                    )
                except EmailNotValidError:
                    continue
        return results

    # ---------------- DOMAIN NORMALIZATION ----------------

    @lru_cache(maxsize=256)
    def _normalize_domain(self, full_domain):
        domain_lower = full_domain.lower()

        if domain_lower in self.high_confidence_replacements:
            return self.high_confidence_replacements[domain_lower]

        best_match_domain = full_domain
        best_score = 0

        for known in self.known_domains:
            score = fuzz.ratio(domain_lower, known.lower())
            if score > best_score:
                best_score = score
                best_match_domain = known

        return (
            best_match_domain
            if best_score >= self.domain_correction_threshold
            else full_domain
        )

    # ---------------- DEDUPLICATION ----------------

    def _deduplicate_emails(self, raw_results):
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

            corrected_domain = self._normalize_domain(full_domain)
            email = f"{local_part}@{corrected_domain}"
            local, domain = split_email(email)
            if not local or not domain:
                continue

            found = False
            for existing in emails:
                existing_local, existing_domain = split_email(existing)
                local_match = fuzz.ratio(local, existing_local) >= self.local_part_ratio
                domain_match = (
                    fuzz.ratio(domain, existing_domain) >= self.domain_part_ratio
                )
                if local_match and domain_match:
                    group = emails[existing]
                    group["frames_seen"].add(entry["frame"])
                    group["confidences"].append(entry["conf"])
                    group["last_seen_sec"] = max(group["last_seen_sec"], entry["sec"])
                    found = True
                    break

            if not found:
                emails[email] = {
                    "frames_seen": {entry["frame"]},
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

    # ---------------- SAVE OUTPUTS ----------------

    def _save_to_csv(self, records, filename, video_name):
        df = pd.DataFrame(records)
        if not df.empty:
            df["video"] = video_name
            df["first_seen_time"] = df["first_seen_sec"].apply(
                lambda x: str(datetime.timedelta(seconds=round(x)))
            )
            df["last_seen_time"] = df["last_seen_sec"].apply(
                lambda x: str(datetime.timedelta(seconds=round(x)))
            )
        df.to_csv(filename, index=False)

    def _save_to_json(self, records, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

    def _cleanup_frames(self):
        shutil.rmtree(self.frames_dir, ignore_errors=True)

    # ---------------- MAIN PIPELINE ----------------

    def run(self):
        video_name = os.path.basename(self.video_path)
        print(f"--- Processing {video_name} ---")
        print(datetime.datetime.now())

        print("\nExtracting frames...")
        self._extract_frames()

        print("\nSelecting unique frames...")
        unique = self._unique_frames(True)
        print(f"{len(unique)} unique frames retained.")

        print("\nRunning OCR and extracting emails...")
        results = self._extract_emails_from_frames(unique)
        print(f"Detected {len(results)} raw email instances.")

        print("\nDeduplicating results...")
        deduped = self._deduplicate_emails(results)
        print(f"{len(deduped)} unique emails after deduplication.")

        csv_file = "output.csv"
        json_file = "output.json"

        print("\nSaving outputs...")
        self._save_to_csv(deduped, csv_file, video_name)
        self._save_to_json(deduped, json_file)

        if not self.keep_frames:
            print("\nCleaning up frames...")
            self._cleanup_frames()

        print("\nDone! Output saved to:")
        print(f"  - {csv_file}")
        print(f"  - {json_file}")
        print(datetime.datetime.now())


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    processor = VideoProcessor(
        video_path="input.mp4",
        # video_path="Screencast from 2025-10-16 15-16-43.webm",
        frame_fps=10,
        use_gpu=True,
        roi=None,  # e.g. (100, 500, 200, 900)
        keep_frames=False,
    )
    processor.run()
