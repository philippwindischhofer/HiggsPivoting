import os, psutil, time
import subprocess as sp

class LocalJobSubmitter:

    @staticmethod
    def submit_job(job_script_path):

        while True:
            time.sleep(1)
            cur_utilization = psutil.cpu_percent()
            if cur_utilization < 90:
                break

        try:
            #sp.check_output(["sh", job_script_path])
            sp.Popen(["sh", job_script_path])
            print("launched job")
        except:
            print("Error in child process caught and ignored.")
