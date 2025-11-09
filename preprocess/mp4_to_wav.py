import os
import subprocess

folder = "MELD.Raw/output_repeated_splits_test"
out_folder = "MELD.Raw/test_audio"

for filename in os.listdir(folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(folder, filename)
        output_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(out_folder, output_name)

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vn",
            "-acodec", "pcm_s16le",
            output_path
        ]
        subprocess.run(cmd)