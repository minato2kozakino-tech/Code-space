import os
import subprocess
import time

"""
script will write htop program to log file and clear this in every 5 seconds no output, use subocess to run htop command and redirect the output to a log file. Then, use a loop to check the log file every 5 seconds and clear it if there is no output.
"""
def run_htop():
    log_file = "htop_log.txt"
    
    # Run htop command and redirect output to log file
    with open(log_file, "w") as f:
        subprocess.Popen(["htop"], stdout=f, stderr=subprocess.STDOUT)
    
    while True:
        # Check if log file is empty
        if os.path.getsize(log_file) == 0:
            print("No output from htop, clearing log file.")
            open(log_file, "w").close()  # Clear the log file
        else:
            print("Output detected in log file.")
        
        time.sleep(5)  # Wait for 5 seconds before checking again

if __name__ == "__main__":
    run_htop()