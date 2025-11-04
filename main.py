import os
import sys
from video_processor import VideoProcessor

if __name__ == "__main__":
    video_name = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    video_path = os.path.join(os.getcwd(), video_name)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    print(f"🔍 Processing video: {video_path}")

    processor = VideoProcessor(
        video_path=video_path,
        frame_fps=10,
        use_gpu=True,
        is_lower_case=True,
        roi=None,
        phash_threshold=0,
        local_part_ratio=90,
        domain_part_ratio=70,
        domain_correction_threshold=80,
        max_segment_seconds=300,
        gpu_per_ocr_mb=600,
    )

    results = processor.run()

    os.makedirs("outputs", exist_ok=True)

    csv_path = os.path.join("outputs", f"{os.path.splitext(video_name)[0]}_results.csv")
    json_path = os.path.join(
        "outputs", f"{os.path.splitext(video_name)[0]}_results.json"
    )

    processor._save_to_csv(results, csv_path, video_name)
    processor._save_to_json(results, json_path)

    print("\n✅ Done! Results saved to:")
    print(f"   {csv_path}")
    print(f"   {json_path}")
