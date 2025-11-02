import os
import cv2
import re
import subprocess
import shutil
import datetime
import math
import tempfile
import json
import imagehash
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from rapidfuzz import fuzz
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"


def show_image(title, img, cmap="gray", export=False, show=False, ocr_res=None):
    if ocr_res and len(ocr_res) and ocr_res[0] is not None:
        for line in ocr_res[0]:
            box, (text, conf) = line
            box = np.array(box).astype(int)

            cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=1)

            if re.search(EMAIL_REGEX, text):
                x, y = int(box[0][0]), int(box[0][1]) - 5
                cv2.putText(
                    img,
                    f"{text}-{conf:.3f}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    if export and title:
        cv2.imwrite(title, img)
    if show:
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
        plt.show()


def assess_quality(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.ndim == 2:
        gray = image
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        if np.mean(image[:, :, 0]) > np.mean(image[:, :, 2]):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format")

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur_score / 1000, 1.0)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bright_pixels = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    glare_ratio = np.sum(bright_pixels > 0) / (image_bgr.shape[0] * image_bgr.shape[1])
    glare_score = 1.0 - min(glare_ratio * 2, 1.0)

    corners = [
        image_bgr[: image_bgr.shape[0] // 4, : image_bgr.shape[1] // 4],
        image_bgr[: image_bgr.shape[0] // 4, 3 * image_bgr.shape[1] // 4 :],
        image_bgr[3 * image_bgr.shape[0] // 4 :, : image_bgr.shape[1] // 4],
        image_bgr[3 * image_bgr.shape[0] // 4 :, 3 * image_bgr.shape[1] // 4 :],
    ]

    dark_corners = 0
    for corner in corners:
        if corner.size > 0:
            mean_brightness = np.mean(cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY))
            if mean_brightness < 50:
                dark_corners += 1
    completeness_score = 1.0 - (dark_corners / 4.0)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()

    under_exposed = np.sum(hist[:50])
    over_exposed = np.sum(hist[200:])
    exposure_score = 1.0 - (under_exposed + over_exposed)
    exposure_score = max(0, min(1, exposure_score))

    overall_score = (
        blur_score + glare_score + completeness_score + exposure_score
    ) / 4.0

    return {
        "blur_score": blur_score,
        "glare_score": glare_score,
        "completeness_score": completeness_score,
        "exposure_score": exposure_score,
        "overall_score": overall_score,
    }


class VideoProcessor:
    DEFAULT_FRAME_FPS = 5
    DEFAULT_PHASH_THRESHOLD = 3
    DEFAULT_LOCAL_PART_RATIO = 90
    DEFAULT_DOMAIN_PART_RATIO = 70
    DEFAULT_DOMAIN_CORRECTION_THRESHOLD = 90

    DEFAULT_MAX_SEGMENT_SECONDS = 300

    DEFAULT_GPU_PER_OCR_MB = 600

    GPU_RESERVE_MB = 500

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

    def __init__(
        self,
        video_path,
        frames_dir=None,
        frame_fps: int = None,
        use_gpu: bool = True,
        roi: tuple = None,
        keep_frames: bool = False,
        phash_threshold: int = None,
        local_part_ratio: int = None,
        domain_part_ratio: int = None,
        domain_correction_threshold: int = None,
        max_segment_seconds: int = None,
        gpu_per_ocr_mb: int = None,
        high_confidence_replacements: dict = None,
        known_domains: list = None,
    ):
        self.video_path = video_path
        self.frames_dir = frames_dir or ""
        self.frame_fps = frame_fps or self.DEFAULT_FRAME_FPS
        self.use_gpu = use_gpu
        self.roi = roi
        self.keep_frames = keep_frames
        self.phash_threshold = phash_threshold or self.DEFAULT_PHASH_THRESHOLD
        self.local_part_ratio = local_part_ratio or self.DEFAULT_LOCAL_PART_RATIO
        self.domain_part_ratio = domain_part_ratio or self.DEFAULT_DOMAIN_PART_RATIO
        self.domain_correction_threshold = (
            domain_correction_threshold or self.DEFAULT_DOMAIN_CORRECTION_THRESHOLD
        )
        self.max_segment_seconds = (
            max_segment_seconds or self.DEFAULT_MAX_SEGMENT_SECONDS
        )
        self.gpu_per_ocr_mb = gpu_per_ocr_mb or self.DEFAULT_GPU_PER_OCR_MB

        self.high_confidence_replacements = (
            high_confidence_replacements or self.HIGH_CONFIDENCE_REPLACEMENTS
        )
        self.known_domains = known_domains or list(self.KNOWN_DOMAINS)

        self.csv_file = "output.csv"
        self.json_file = "output.json"

    @staticmethod
    def _run_cmd(cmd):
        return subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def _get_video_duration(self, path):
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                stderr=subprocess.DEVNULL,
            )
            return float(out.strip())
        except Exception:
            return None

    def _query_gpu_free_mb(self):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )

            lines = out.decode("utf-8").strip().splitlines()
            frees = [int(l.strip()) for l in lines if l.strip().isdigit()]
            if not frees:
                return None

            return max(frees)
        except Exception:
            return None

    def _decide_worker_count(self, requested_max=None):
        cpu_count = multiprocessing.cpu_count()
        max_by_cpu = min(4, cpu_count)

        if not self.use_gpu:
            return requested_max or max_by_cpu

        free_mb = self._query_gpu_free_mb()
        if free_mb is None:
            return requested_max or max_by_cpu

        effective_free = max(0, free_mb - self.GPU_RESERVE_MB)
        possible = max(1, effective_free // self.gpu_per_ocr_mb)

        return min(possible, cpu_count, requested_max or possible or 1)

    def _extract_frames(self, video_file, frames_dir):
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_file,
                "-vf",
                f"fps={self.frame_fps}",
                os.path.join(frames_dir, "frame_%06d.png"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _compute_phash(self, path):
        with Image.open(path) as img:
            return imagehash.phash(img)

    def _unique_frames_progressive(self, frames_dir):
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        if not frames:
            return []

        hashes = []
        unique = []
        for fname in tqdm(frames, desc="Filtering unique frames"):
            path = os.path.join(frames_dir, fname)
            with Image.open(path) as img:
                phash = imagehash.phash(img)
                qc = assess_quality(img)
                hashes.append({"fname": fname, "phash": phash, "qc": qc})

        i = 0
        last_qc = 0
        for hash in hashes:
            last = hashes[i - 1] if len(hashes) else hash
            fname = hash["fname"]
            if last["phash"] - hash["phash"] == 0:
                if hash["qc"]["overall_score"] - last_qc > 0:
                    last_qc = hash["qc"]["overall_score"]
                    unique.append(fname)
            else:
                unique.append(fname)
                last_qc = hash["qc"]["overall_score"]
            i += 1

        return unique

    def _crop_roi(self, image):
        if self.roi is None:
            return image
        y0, y1, x0, x1 = self.roi
        return image[y0:y1, x0:x1]

    def _keep_dark_regions(self, img, threshold=200):
        img = np.asarray(img, dtype=np.uint8)

        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(
            np.uint8
        )

        mask = gray < threshold

        result = np.ones_like(img) * 255

        result[mask] = np.stack([gray[mask]] * 3, axis=-1)
        return result

    def _extract_emails_from_frames(self, frames_dir, unique_frames):
        ocr = None
        ocr = PaddleOCR(
            use_angle_cls=True, lang="en", use_gpu=self.use_gpu, show_log=False
        )
        results = []

        for idx, fname in enumerate(tqdm(unique_frames, desc="Running OCR")):
            path = os.path.join(frames_dir, fname)
            img_cv = cv2.imread(path)
            if img_cv is None:
                continue
            if self.roi:
                img_cv = self._crop_roi(img_cv)
            img_cv = self._keep_dark_regions(img_cv)

            ocr_res = ocr.ocr(img_cv, cls=True)
            if not ocr_res or not ocr_res[0]:
                continue
            show_image(f"frames-out/{fname}", img_cv, export=True)

            for line in ocr_res[0]:
                box, (text, conf) = line
                emails = re.findall(EMAIL_REGEX, text)
                for email in emails:
                    email = email.replace("I", "l")
                    results.append(
                        {
                            "email": email.strip(),
                            "frame": fname,
                            "conf": conf,
                            "sec": idx / self.frame_fps,
                            "box": box,
                        }
                    )

        return results

    @lru_cache(maxsize=512)
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
            for existing in list(emails.keys()):
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
                    group["first_seen_sec"] = min(group["first_seen_sec"], entry["sec"])
                    group["sample_frame"] = group.get("sample_frame", entry["frame"])
                    group["box"] = group.get("box", entry["box"])
                    found = True
                    break

            if not found:
                emails[email] = {
                    "frames_seen": {entry["frame"]},
                    "confidences": [entry["conf"]],
                    "first_seen_sec": entry["sec"],
                    "last_seen_sec": entry["sec"],
                    "sample_frame": entry["frame"],
                    "box": entry["box"],
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
                    "box": data["box"],
                }
            )
        return final

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

    def _split_video_into_segments(self, video_path, out_dir, segment_seconds):
        duration = self._get_video_duration(video_path)
        if duration is None:
            return [video_path]

        if duration <= segment_seconds:
            return [video_path]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        original_ext = os.path.splitext(video_path)[1].lower()
        if not original_ext:
            original_ext = ".mp4"

        segment_paths = []
        num_segments = math.ceil(duration / segment_seconds)
        for i in range(num_segments):
            start = i * segment_seconds
            seg_path = os.path.join(out_dir, f"segment_{i:03d}.{original_ext}")

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    video_path,
                    "-t",
                    str(segment_seconds),
                    "-c",
                    "copy",
                    seg_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            segment_paths.append(seg_path)
        return segment_paths

    def _process_segment_return_raw(self, segment_path, segment_index, temp_root):
        seg_frames_dir = os.path.join(
            self.frames_dir or temp_root, f"frames_seg_{segment_index:03d}"
        )

        seg_proc = VideoProcessor(
            video_path=segment_path,
            frames_dir=seg_frames_dir,
            frame_fps=self.frame_fps,
            use_gpu=self.use_gpu,
            roi=self.roi,
            keep_frames=self.keep_frames,
            phash_threshold=self.phash_threshold,
            local_part_ratio=self.local_part_ratio,
            domain_part_ratio=self.domain_part_ratio,
            domain_correction_threshold=self.domain_correction_threshold,
            max_segment_seconds=self.max_segment_seconds,
            gpu_per_ocr_mb=self.gpu_per_ocr_mb,
            high_confidence_replacements=self.high_confidence_replacements,
            known_domains=self.known_domains,
        )

        seg_proc._extract_frames(segment_path, seg_frames_dir)

        unique_frames = seg_proc._unique_frames_progressive(seg_frames_dir)

        raw_results = seg_proc._extract_emails_from_frames(
            seg_frames_dir, unique_frames
        )

        if not seg_proc.keep_frames:
            shutil.rmtree(seg_frames_dir, ignore_errors=True)
        return raw_results

    def run(self):
        video_name = os.path.basename(self.video_path)
        print(f"--- Processing {video_name} ---")
        start_time = datetime.datetime.now()
        print("start:", start_time)

        duration = self._get_video_duration(self.video_path)
        if duration is None:
            print("Warning: couldn't detect duration; will treat as single segment.")
        print(f"video length: {duration}s")

        segment_seconds = self.max_segment_seconds
        if duration is None or duration <= segment_seconds:
            temp_root = tempfile.mkdtemp(prefix="vp_single_")
            try:
                raw = self._process_segment_return_raw(self.video_path, 0, temp_root)
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
            combined_raw = raw
        else:
            print(
                f"Video duration: {duration:.1f}s, splitting into {segment_seconds}s segments."
            )
            segments_dir = tempfile.mkdtemp(prefix="vp_segments_")
            try:
                segment_files = self._split_video_into_segments(
                    self.video_path, segments_dir, segment_seconds
                )
                print(f"Created {len(segment_files)} segment files.")

                worker_count = self._decide_worker_count(
                    requested_max=len(segment_files)
                )
                worker_count = max(1, min(worker_count, len(segment_files)))
                print(
                    f"Processing segments with up to {worker_count} parallel workers (use_gpu={self.use_gpu})."
                )

                temp_root = tempfile.mkdtemp(prefix="vp_seg_frames_")
                all_raw = []

                with ThreadPoolExecutor(max_workers=worker_count) as ex:
                    futures = []
                    for idx, seg in enumerate(segment_files):
                        futures.append(
                            ex.submit(
                                self._process_segment_return_raw, seg, idx, temp_root
                            )
                        )

                    for fut in tqdm(
                        as_completed(futures), total=len(futures), desc="Segments done"
                    ):
                        try:
                            partial = fut.result()
                            all_raw.extend(partial)
                        except Exception as e:
                            print("Segment processing error:", str(e))

                combined_raw = all_raw
            finally:
                shutil.rmtree(segments_dir, ignore_errors=True)
                try:
                    shutil.rmtree(temp_root, ignore_errors=True)
                except Exception:
                    pass

        print(
            f"Collected {len(combined_raw)} raw candidate email instances across segments."
        )
        print("Deduplicating globally...")
        deduped = self._deduplicate_emails(combined_raw)
        print(f"{len(deduped)} unique emails after deduplication.")

        print("Saving outputs...")
        self._save_to_csv(deduped, self.csv_file, video_name)
        self._save_to_json(deduped, self.json_file)

        if os.path.exists(self.frames_dir) and not self.keep_frames:
            try:
                shutil.rmtree(self.frames_dir, ignore_errors=True)
            except Exception:
                pass

        end_time = datetime.datetime.now()
        print("Done. Outputs:", self.csv_file, self.json_file)
        print("start:", start_time, "end:", end_time, "elapsed:", end_time - start_time)
        return deduped


# ------------------ Example usage ------------------

if __name__ == "__main__":
    processor = VideoProcessor(
        video_path="Screencast from 2025-10-16 15-16-43.webm",
        frame_fps=10,
        use_gpu=True,
        roi=None,
        keep_frames=True,
        frames_dir="frames",
        phash_threshold=0,
        local_part_ratio=90,
        domain_part_ratio=70,
        domain_correction_threshold=90,
        max_segment_seconds=300,
        gpu_per_ocr_mb=600,
    )

    results = processor.run()
