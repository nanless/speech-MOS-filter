import subprocess
import time

def run_script(index=9):
    # Replace 'your_script.py' with the actual name of your script and add necessary command line arguments
    
    command = ["python", "compute_mos.py", "--wav_files_dir", "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/clean_fullband", "--output_statistics_dir", f"/root/autodl-tmp/DNS_challenge5_data/clean_mos_metrics_{index}", "--batch_size", "200", "--previous_statistics_dirs", "/root/autodl-tmp/DNS_challenge5_data/clean_mos_metrics"]
    for i in range(2, index):
        command.append(f"/root/autodl-tmp/DNS_challenge5_data/clean_mos_metrics_{i}")
    # Run the script
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def main():

    cur_index = 9

    while True:

        process = run_script(cur_index)

        # Wait for the script to finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            # If the script crashed, log the error and restart it
            print("Script crashed")
            print("Error output:", stderr.decode())
            print("Restarting script...")
            time.sleep(1)  # sleep for a bit before restarting to avoid spamming
            cur_index += 1
        else:
            # If the script finishes successfully, exit the loop
            print("Script finished successfully")
            break

if __name__ == "__main__":
    main()
