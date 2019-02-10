import os
import subprocess as sp

class CondorJobSubmitter:

    # create the .submit file and submit it to the batch
    @staticmethod
    def submit_job(job_script_path):
        job_script_base, _ = os.path.splitext(job_script_path)
        job_dir = os.path.dirname(job_script_path)
        submit_file_path = job_script_base + ".submit"

        with open(submit_file_path, 'w') as submit_file:
            submit_file.write("executable = " + job_script_path + "\n")
            submit_file.write("universe = vanilla\n")
            submit_file.write("output = " + os.path.join(job_dir, "output.$(Process)\n"))
            submit_file.write("error = " + os.path.join(job_dir, "error.$(Process)\n"))
            submit_file.write("log = " + os.path.join(job_dir, "log.$(Process)\n"))
            submit_file.write("notification = never\n")
            submit_file.write("request_cpus = 1\n")
            # submit_file.write("request_memory = 4000\n") # in MB
            # submit_file.write("should_transfer_files = Yes\n")
            # submit_file.write("when_to_transfer_output = ON_EXIT\n")
            submit_file.write("queue 1")

        # call the job submitter
        sp.check_output(["condor_submit", submit_file_path])
        print("submitted '" + submit_file_path + "'")
