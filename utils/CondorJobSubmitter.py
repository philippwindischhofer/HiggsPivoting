import os, re, time
import subprocess as sp

class CondorJobSubmitter:

    # create the .submit file and submit it to the batch
    @staticmethod
    def submit_job(job_script_path, job_threshold = 400):
        job_script_base, _ = os.path.splitext(job_script_path)
        job_dir = os.path.dirname(job_script_path)
        submit_file_path = job_script_base + ".submit"

        with open(submit_file_path, 'w') as submit_file:
            submit_file.write("executable = " + job_script_path + "\n")
            submit_file.write("universe = vanilla\n")
            submit_file.write("output = output.$(Process)\n")
            submit_file.write("error = error.$(Process)\n")
            submit_file.write("log = log.$(Process)\n")
            submit_file.write("notification = never\n")
            submit_file.write("request_cpus = 8\n")
            submit_file.write("request_memory = 10000\n") # in MB
            # submit_file.write("should_transfer_files = Yes\n")
            # submit_file.write("when_to_transfer_output = ON_EXIT\n")
            submit_file.write("queue 1")

        while True:
            running_jobs = CondorJobSubmitter.queued_jobs()
            if running_jobs < job_threshold:
                break
            print("have {} jobs running - wait a bit".format(running_jobs))
            time.sleep(30)

        while True:
            try:
                # call the job submitter
                sp.check_output(["condor_submit", submit_file_path])
                print("submitted '" + submit_file_path + "'")
                break
            except:
                print("problem with submitter -- retrying")
                time.sleep(10)

    @staticmethod
    def queued_jobs(queue_status = "condor_q"):
        while True:
            try:
                running_jobs = len(sp.check_output([queue_status]).decode("utf8").split('\n')) - 6
                return running_jobs
            except sp.CalledProcessError:
                print("{} error - retrying!".format(queue_status))
                time.sleep(10)

    @staticmethod
    def get_running_cluster_IDs():

        while True:
            try:
                running = sp.check_output(["condor_q", "-long", "-af", "JOB_IDS"]).decode("utf-8").split('\n')
                break
            except:
                print("problem with job lister -- retrying")
                time.sleep(1)
       
        def cluster_extractor(instring):
            job_id_finder = re.compile("ClusterId\\s*=\\s*(.+)")
            m = job_id_finder.match(instring)
            if m:
                return m.group(1)
            else:
                return None

        cluster_IDs = set(map(cluster_extractor, running))
        cluster_IDs.remove(None)

        return cluster_IDs
