### Project Summary

This project focuses on extracting email addresses from CRM-recording videos and exporting them to an Excel-compatible CSV. It meets the client's goal of high accuracy (95%+) for specific, non-scalable use cases, within a strict timeline. All processing is performed locally, taking input video files from the same directory.

---

### Technical Approach

#### 1. Frame Extraction

- The system uses **FFmpeg** to split the input video into image frames at a fixed rate (2 frames per second).
- This step ensures coverage across the video's scrolling CRM interface and is robust for various video formats.

#### 2. Frame Filtering

- Perceptual hashing (**pHash**) is applied to skip frames that are visually too similar, reducing computational load while preserving email visibility.
- The process is designed to quickly discard essentially duplicate scenes, increasing throughput.

#### 3. Optional ROI Cropping

- You can define a region of interest to focus only on CRM columns or areas where emails are displayed, using **OpenCV**.

#### 4. OCR for Text Extraction

- **PaddleOCR** is run on each unique frame to detect and extract on-screen text.
- The tool excels at recognizing Latin-character email addresses, and outputs text along with a confidence score per detection.

#### 5. Email Parsing and Validation

- A regex pattern is used to extract valid email formats from the OCR results.
- Each found email is checked for validity using the `email-validator` library, filtering out most false positives.

#### 6. Deduplication

- Extracted emails are grouped using **RapidFuzz** fuzzy matching, merging near-identical emails that result from OCR errors.

#### 7. Reporting & Metrics

- Results are written to CSV, including:  
  `email, frames_seen, mean_conf, first_seen_sec, last_seen_sec, sample_frame`
- All intermediate frames are deleted after extraction for privacy.
- Detailed logging and CSV contents enable the client to manually review performance as needed.

### installation

- using gpu

```
pip install "paddleocr==2.7.0.3" "paddlepaddle-gpu==2.6.2" "opencv-python==4.10.0.84" "shapely==2.0.5" "pyclipper==1.3.0.post5"
```

- easyocr

```
pip install numpy==1.26.4
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install easyocr==1.7.2

pip cache purge
pip install numpy==1.26.4 --force-reinstall
```
