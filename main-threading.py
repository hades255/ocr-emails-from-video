import os
import cv2
import re
import subprocess
import shutil
import datetime
import math
import tempfile
from PIL import Image
import imagehash
from paddleocr import PaddleOCR
import pandas as pd
from rapidfuzz import fuzz
# email validation is optional (costly) — uncomment if you want stricter validation:
# from email_validator import validate_email, EmailNotValidError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import lru_cache
import json
import multiprocessing


class VideoProcessor:
    """
    VideoProcessor: extracts emails from a video using frame extraction, pHash dedup,
    PaddleOCR and fuzzy dedup across frames.

    This class supports splitting long videos into segments and multi-threaded processing
    while taking GPU memory into account to avoid overcommitting GPU resources.
    """

    # ------------------ DEFAULT CONFIG ------------------
    # Defaults tuned for moderate-length CRM screen recordings.
    DEFAULT_FRAME_FPS = 5  # frames per second extracted from video
    DEFAULT_PHASH_THRESHOLD = 3
    DEFAULT_LOCAL_PART_RATIO = 90
    DEFAULT_DOMAIN_PART_RATIO = 70
    DEFAULT_DOMAIN_CORRECTION_THRESHOLD = 90

    # If segment length (seconds) exceeds this, video will be split.
    # Default: 5 minutes. You can lower for shorter segments or raise for fewer segments.
    DEFAULT_MAX_SEGMENT_SECONDS = 300

    # GPU memory we assume a PaddleOCR instance will occupy (MB). Tune this per your GPU.
    # PaddleOCR baseline ~ 400-800MB depending on model/version - 600MB conservative default.
    DEFAULT_GPU_PER_OCR_MB = 600

    # Minimal free GPU memory to keep available for system / other processes (MB).
    GPU_RESERVE_MB = 500

    EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

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

    # ------------------ INIT ------------------

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
        self.frames_dir = frames_dir or tempfile.mkdtemp(prefix="vp_frames_")
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
        self.max_segment_seconds = max_segment_seconds or self.DEFAULT_MAX_SEGMENT_SECONDS
        self.gpu_per_ocr_mb = gpu_per_ocr_mb or self.DEFAULT_GPU_PER_OCR_MB

        self.email_regex = self.EMAIL_REGEX
        self.high_confidence_replacements = (
            high_confidence_replacements or self.HIGH_CONFIDENCE_REPLACEMENTS
        )
        self.known_domains = known_domains or list(self.KNOWN_DOMAINS)

        # Internal outputs
        self.csv_file = "output.csv"
        self.json_file = "output.json"

    # ------------------ Utilities ------------------

    @staticmethod
    def _run_cmd(cmd):
        """Run a system command and return (stdout, stderr). Raises on non-zero."""
        return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _get_video_duration(self, path):
        """Return video duration in seconds using ffprobe."""
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
            # fallback: unknown duration -> treat as large so splitting may not occur
            return None

    # ------------------ GPU helpers ------------------

    def _query_gpu_free_mb(self):
        """
        Query the first GPU's free memory using nvidia-smi.
        Returns free_mb (int) or None if nvidia-smi not available.
        """
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            # out can contain multiple lines (one per GPU). We'll use the largest free.
            lines = out.decode("utf-8").strip().splitlines()
            frees = [int(l.strip()) for l in lines if l.strip().isdigit()]
            if not frees:
                return None
            # return total free across GPUs or max — choose max
            return max(frees)
        except Exception:
            return None

    def _decide_worker_count(self, requested_max=None):
        """
        Decide how many parallel segment workers to run.
          - If GPU available and use_gpu=True: limit by free memory and per-OCR estimate.
          - Else: limit by CPU count and a conservative cap.
        """
        cpu_count = multiprocessing.cpu_count()
        max_by_cpu = min(4, cpu_count)  # don't spawn too many CPU OCR workers by default

        if not self.use_gpu:
            return requested_max or max_by_cpu

        free_mb = self._query_gpu_free_mb()
        if free_mb is None:
            # couldn't detect GPU; fall back to CPU-based
            return requested_max or max_by_cpu

        effective_free = max(0, free_mb - self.GPU_RESERVE_MB)
        possible = max(1, effective_free // self.gpu_per_ocr_mb)
        # never exceed CPU capacity either
        return min(possible, cpu_count, requested_max or possible or 1)

    # ------------------ Frame extraction ------------------

    def _extract_frames(self, video_file, frames_dir):
        """Extract frames from a video file at configured FPS into frames_dir."""
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

    # ------------------ pHash uniqueness ------------------

    def _compute_phash(self, path):
        with Image.open(path) as img:
            return imagehash.phash(img)

    def _unique_frames_progressive(self, frames_dir):
        """Original progressive behavior: keep frame if different from ALL previously kept frames."""
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        if not frames:
            return []

        hashes = []
        unique = []
        for fname in tqdm(frames, desc="Filtering unique frames"):
            path = os.path.join(frames_dir, fname)
            with Image.open(path) as img:
                phash = imagehash.phash(img)
            if all(abs(phash - h) > self.phash_threshold for h in hashes):
                hashes.append(phash)
                unique.append(fname)
        return unique

    # ------------------ OCR & per-frame extraction ------------------

    def _crop_roi(self, image):
        if self.roi is None:
            return image
        y0, y1, x0, x1 = self.roi
        return image[y0:y1, x0:x1]

    def _extract_emails_from_frames(self, frames_dir, unique_frames):
        """
        Run PaddleOCR and extract candidate emails (raw results).
        Returns list of dicts: {email, frame, conf, sec}
        """
        # Each thread / worker gets its own OCR instance
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self.use_gpu, show_log=False)
        results = []

        for idx, fname in enumerate(tqdm(unique_frames, desc="Running OCR")):
            path = os.path.join(frames_dir, fname)
            img_cv = cv2.imread(path)
            if img_cv is None:
                continue
            if self.roi:
                img_cv = self._crop_roi(img_cv)

            # PaddleOCR accepts either image path or numpy image
            ocr_res = ocr.ocr(img_cv, cls=True)
            if not ocr_res or not ocr_res[0]:
                continue

            # extract text segments with confidence > 0.5
            frame_text_pieces = [line[1][0] for line in ocr_res[0] if line[1][1] > 0.5]
            frame_text = " ".join(frame_text_pieces)

            for email in re.findall(self.email_regex, frame_text):
                # skip calling validate_email by default for speed; uncomment if desired
                try:
                    conf = [line[1][1] for line in ocr_res[0] if email in line[1][0]]
                    results.append(
                        {
                            "email": email,
                            "frame": fname,
                            "conf": sum(conf) / len(conf) if conf else 0.5,
                            "sec": idx / self.frame_fps,
                        }
                    )
                except Exception:
                    # If validation used, catch EmailNotValidError here
                    continue
        return results

    # ------------------ Domain normalization ------------------

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

        return best_match_domain if best_score >= self.domain_correction_threshold else full_domain

    # ------------------ Deduplication (global) ------------------

    def _deduplicate_emails(self, raw_results):
        """Global fuzzy dedup across combined raw results from all segments."""
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
                domain_match = fuzz.ratio(domain, existing_domain) >= self.domain_part_ratio
                if local_match and domain_match:
                    group = emails[existing]
                    group["frames_seen"].add(entry["frame"])
                    group["confidences"].append(entry["conf"])
                    group["last_seen_sec"] = max(group["last_seen_sec"], entry["sec"])
                    group["first_seen_sec"] = min(group["first_seen_sec"], entry["sec"])
                    group["sample_frame"] = group.get("sample_frame", entry["frame"])
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
                sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0.0
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

    # ------------------ Save outputs ------------------

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

    # ------------------ Segment splitting ------------------

    def _split_video_into_segments(self, video_path, out_dir, segment_seconds):
        """Split the input video into sequential segments (using ffmpeg -ss/-t)."""
        duration = self._get_video_duration(video_path)
        if duration is None:
            # If unknown, treat as single segment
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
            # Use -ss before -i to be faster (but less precise); here we use it with -c copy for speed.
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

    # ------------------ Per-segment worker (returns raw results) ------------------

    def _process_segment_return_raw(self, segment_path, segment_index, temp_root):
        """
        Process a single segment and return raw (non-deduplicated) results list.
        Uses its own frames_dir so parallelism is safe.
        """
        seg_frames_dir = os.path.join(temp_root, f"frames_seg_{segment_index:03d}")
        # instantiate a processor for the segment with same configuration but unique frames dir
        seg_proc = VideoProcessor(
            video_path=segment_path,
            frames_dir=seg_frames_dir,
            frame_fps=self.frame_fps,
            use_gpu=self.use_gpu,
            roi=self.roi,
            keep_frames=self.keep_frames,  # per-worker decide
            phash_threshold=self.phash_threshold,
            local_part_ratio=self.local_part_ratio,
            domain_part_ratio=self.domain_part_ratio,
            domain_correction_threshold=self.domain_correction_threshold,
            max_segment_seconds=self.max_segment_seconds,
            gpu_per_ocr_mb=self.gpu_per_ocr_mb,
            high_confidence_replacements=self.high_confidence_replacements,
            known_domains=self.known_domains,
        )

        # 1) extract frames for this segment
        seg_proc._extract_frames(segment_path, seg_frames_dir)

        # 2) pick unique frames progressively (keeps more frames like original)
        unique_frames = seg_proc._unique_frames_progressive(seg_frames_dir)

        # 3) OCR and collect raw results for this segment
        raw_results = seg_proc._extract_emails_from_frames(seg_frames_dir, unique_frames)

        # 4) cleanup frames for this segment unless keep_frames True
        if not seg_proc.keep_frames:
            shutil.rmtree(seg_frames_dir, ignore_errors=True)
        return raw_results

    # ------------------ Main runner ------------------

    def run(self):
        """
        High-level runner:
          - checks video duration
          - if exceeds max_segment_seconds, splits into segments
          - decides worker count based on GPU memory/CPU
          - processes segments (multi-threaded)
          - combines raw results and deduplicates globally
          - saves CSV/JSON
        """
        video_name = os.path.basename(self.video_path)
        print(f"--- Processing {video_name} ---")
        start_time = datetime.datetime.now()
        print("start:", start_time)

        duration = self._get_video_duration(self.video_path)
        if duration is None:
            print("Warning: couldn't detect duration; will treat as single segment.")

        # Decide segmentation
        segment_seconds = self.max_segment_seconds
        if duration is None or duration <= segment_seconds:
            # Single segment: process locally and dedupe
            temp_root = tempfile.mkdtemp(prefix="vp_single_")
            try:
                raw = self._process_segment_return_raw(self.video_path, 0, temp_root)
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
            combined_raw = raw
        else:
            # Split and parallel process
            print(f"Video duration: {duration:.1f}s, splitting into {segment_seconds}s segments.")
            segments_dir = tempfile.mkdtemp(prefix="vp_segments_")
            try:
                segment_files = self._split_video_into_segments(self.video_path, segments_dir, segment_seconds)
                print(f"Created {len(segment_files)} segment files.")

                # Decide worker count (conservative)
                worker_count = self._decide_worker_count(requested_max=len(segment_files))
                worker_count = max(1, min(worker_count, len(segment_files)))
                print(f"Processing segments with up to {worker_count} parallel workers (use_gpu={self.use_gpu}).")

                temp_root = tempfile.mkdtemp(prefix="vp_seg_frames_")
                all_raw = []

                with ThreadPoolExecutor(max_workers=worker_count) as ex:
                    futures = []
                    for idx, seg in enumerate(segment_files):
                        futures.append(ex.submit(self._process_segment_return_raw, seg, idx, temp_root))

                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Segments done"):
                        try:
                            partial = fut.result()
                            all_raw.extend(partial)
                        except Exception as e:
                            print("Segment processing error:", str(e))

                combined_raw = all_raw
            finally:
                # remove segment files dir
                shutil.rmtree(segments_dir, ignore_errors=True)
                try:
                    shutil.rmtree(temp_root, ignore_errors=True)
                except Exception:
                    pass

        print(f"Collected {len(combined_raw)} raw candidate email instances across segments.")
        print("Deduplicating globally...")
        deduped = self._deduplicate_emails(combined_raw)
        print(f"{len(deduped)} unique emails after deduplication.")

        print("Saving outputs...")
        self._save_to_csv(deduped, self.csv_file, video_name)
        self._save_to_json(deduped, self.json_file)

        # optionally cleanup the class-level frames_dir if created and not requested to keep
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
    # Example: tune these values as you need
    processor = VideoProcessor(
        video_path="input.mp4",
        frame_fps=5,  # frames per second extracted
        use_gpu=True,
        roi=None,  # e.g., (100, 500, 200, 900)
        keep_frames=False,
        phash_threshold=3,
        local_part_ratio=90,
        domain_part_ratio=70,
        domain_correction_threshold=90,
        max_segment_seconds=300,  # split every 5 minutes
        gpu_per_ocr_mb=600,  # adjust if you know real memory usage
    )

    results = processor.run()
    # results is a list of dicts; you can also inspect the saved output.csv
