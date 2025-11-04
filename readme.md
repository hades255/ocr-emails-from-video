## Project Summary

This project focuses on extracting email addresses from CRM-recording videos and exporting them to an Excel-compatible CSV. It meets the client's goal of high accuracy (95%+) for specific, non-scalable use cases, within a strict timeline. All processing is performed locally, taking input video files from the same directory.

## How to set up on a Windows PC (CUDA 11.8)

### ✅ Step 1 — Check GPU driver

- Install latest **NVIDIA driver (CUDA 11.8 capable)**.
- Open **Command Prompt** and run:

  ```powershell
  nvidia-smi
  ```

  You should see your GPU list.

---

### ✅ Step 2 — Install Docker Desktop

1. Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Enable:

   - **WSL 2 backend**
   - **“Use the WSL 2 based engine”**

3. Restart Docker Desktop.

---

### ✅ Step 3 — Install NVIDIA Container Toolkit in WSL

Open **PowerShell (Admin)** and run:

```powershell
wsl --install
wsl -l -v
```

Make sure your default WSL distro (usually Ubuntu) is version 2.

Then inside WSL (Ubuntu terminal):

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access from WSL:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

✅ You should see GPU details.

---

### ✅ Step 4 — Run your project

1. **Put your files** (including `recording3.webm`) in a folder, e.g.:

   ```
   C:\Users\<You>\ocr-app\
   ```

2. Open **PowerShell** in that folder:

   ```powershell
   cd C:\Users\<You>\ocr-app
   docker compose build
   ```

3. Run it:

   ```powershell
   docker compose run --rm ocr_app python main.py recording3.webm
   ```

---

### ✅ Step 5 — Retrieve results

After it finishes, check:

```
C:\Users\<You>\ocr-app\outputs\
 ├─ recording3_results.csv
 └─ recording3_results.json
```

These are directly accessible from Windows (mounted through the `volumes:` mapping).

---

### ✅ Step 6 — Process a different video

Just copy a new file (e.g. `ticket2.mp4`) into the same folder and run:

```powershell
docker compose run --rm ocr_app python main.py ticket2.mp4
```

---

### ✅ Step 7 — Test GPU inside container

To confirm Paddle is using the GPU:

```powershell
docker compose run --rm ocr_app python -c "import paddle; print(paddle.device.get_device())"
```

Expected:

```
gpu:0
```

---

## Technical Approach

### 1. Frame Extraction

- The system uses **FFmpeg** to split the input video into image frames at a fixed rate (2 frames per second).
- This step ensures coverage across the video's scrolling CRM interface and is robust for various video formats.

### 2. Frame Filtering

- Perceptual hashing (**pHash**) is applied to skip frames that are visually too similar, reducing computational load while preserving email visibility.
- The process is designed to quickly discard essentially duplicate scenes, increasing throughput.

### 3. Optional ROI Cropping

- You can define a region of interest to focus only on CRM columns or areas where emails are displayed, using **OpenCV**.

### 4. OCR for Text Extraction

- **PaddleOCR** is run on each unique frame to detect and extract on-screen text.
- The tool excels at recognizing Latin-character email addresses, and outputs text along with a confidence score per detection.

### 5. Email Parsing and Validation

- A regex pattern is used to extract valid email formats from the OCR results.
- Each found email is checked for validity using the `email-validator` library, filtering out most false positives.

### 6. Deduplication

- Extracted emails are grouped using **RapidFuzz** fuzzy matching, merging near-identical emails that result from OCR errors.

### 7. Reporting & Metrics

- Results are written to CSV, including:  
  `email, frames_seen, mean_conf, first_seen_sec, last_seen_sec, sample_frame`
- All intermediate frames are deleted after extraction for privacy.
- Detailed logging and CSV contents enable the client to manually review performance as needed.

## installation

- using gpu

```
pip install "paddleocr==2.7.0.3" "paddlepaddle-gpu==2.6.2" "opencv-python==4.10.0.84" "shapely==2.0.5" "pyclipper==1.3.0.post5"
```
