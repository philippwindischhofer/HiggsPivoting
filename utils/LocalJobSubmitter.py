import os
import subprocess as sp

class LocalJobSubmitter:

    @staticmethod
    def submit_job(job_script_path):
        #try:
        sp.check_output(["sh", job_script_path])
        #except:
        #    print("Error in child process caught.")
