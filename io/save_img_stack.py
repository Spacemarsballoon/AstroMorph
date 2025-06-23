import os
import time
import shutil


def watch_and_copy(source_dir, dest_dir=None, interval=10):
    seen_files = set(os.listdir(source_dir))
    print(f"Monitoring '{source_dir}' every {interval} seconds...")

    while True:
        time.sleep(interval)
        current_files = set(os.listdir(source_dir))
        new_files = current_files - seen_files

        for file in new_files:
            full_path = os.path.join(source_dir, file)
            if os.path.isfile(full_path):
                print(f"[+] New file detected: {file}")
                if dest_dir:
                    shutil.copy(full_path, os.path.join(dest_dir, file))
                    print(f"    Copied to: {dest_dir}")

        seen_files = current_files