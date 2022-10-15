#!/usr/bin/env python3
# --------------------------------------------------------------------------- #


import os
import random
import string
import time
from datetime import datetime


def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


class HTCSubmissionTemplate:
    """A class to provide a template/convencience functions for interfacing
    with HTCondor.

    Parameters
    ----------
    python_file : str
        name of python file to be executed by condor (including .py
        extension)
    folder_in_juseless : str or path
        working directory in which condor should execute the jobs
    venv_activation_file : str
        name of bash script to activate a virtual environment
    submit_file : str
        name of submit file (including directory) to generate for condor
    n_cpus : int
        how many cpus to request
    requested_gb : int
        how many GB of RAM to request
    requested_disk : int
        how many GB of diskspace to request
    logs_dir : str or path
        which directory to use for log files
    """

    def __init__(
        self,
        python_file,
        folder_in_juseless=".",
        venv_activation_file=None,
        submit_file="submit_file.submit",
        n_cpus=1,
        requested_gb=5,
        requested_disk=5,
        logs_dir=".",
    ):
        self.python_file = python_file
        self.folder_in_juseless = os.path.abspath(folder_in_juseless)
        self.venv_activation_file = venv_activation_file
        self.submit_file = submit_file
        self.n_cpus = n_cpus
        self.requested_gb = requested_gb
        self.requested_disk = requested_disk
        self.logs_dir = os.path.abspath(logs_dir)
        self.submission_id = id_generator()
        self.submitted = False

        if self.venv_activation_file is not None:
            self.cmd = f"./{self.venv_activation_file} ./{self.python_file}"
        else:
            self.cmd = f"./{self.python_file}"

    def write_condor_settings(self, submit_file=None):
        """Write top settings to condor_submit file that apply for all jobs

        Parameters
        ----------
        submit_file : str
            name of submit file to write. If no name is provided, will default
            to the submit file specified during initialisation

        """

        if submit_file is None:
            submit_file = self.submit_file

        with open(submit_file, "w") as f:

            # providing settings for hpc
            f.write("executable = /usr/bin/bash\n")
            f.write("transfer_executable = False\n")
            f.write(f"initial_dir= {self.folder_in_juseless}\n")
            f.write("universe = vanilla\n")
            f.write("getenv = True\n")
            f.write(f"request_cpus = {self.n_cpus}\n")
            f.write(f"request_memory = {self.requested_gb}GB\n")
            f.write(f"request_disk = {self.requested_disk}GB\n\n")

    def add_job(
        self,
        args=None,
        job_specific_log=None,
        submit_file=None,
        python_file=None,
    ):
        """Add a job to the submit_file

        Parameters
        ----------
        args : list
            list of arguments to give to the job executed by condor
        job_specific_log : str
            string included in log file names to uniquely identify a job
        submit_file : str
            name of submit file on which to add the job. Defaults to the submit
            file provided during initialisation
        python_file : str
            name of the python file to execute for this job. Defaults to python
            file specified during intialisation

        """

        if submit_file is None:
            submit_file = self.submit_file

        if job_specific_log is not None:
            log_id = f"{self.submission_id}_{job_specific_log}"
        else:
            log_id = self.submission_id

        log_file = os.path.join(
            self.logs_dir, f"{log_id}_$(Cluster).$(Process).log"
        )
        out_file = os.path.join(
            self.logs_dir, f"{log_id}_$(Cluster).$(Process).out"
        )
        err_file = os.path.join(
            self.logs_dir, f"{log_id}_$(Cluster).$(Process).err"
        )

        with open(submit_file, "a") as f:

            if python_file is None:
                cmd = self.cmd
            elif self.venv_activation_file is None:
                cmd = f"./{python_file}"
            else:
                cmd = f"./{self.venv_activation_file} ./{python_file}"

            if args is not None:
                for arg in args:
                    cmd = f"{cmd} {arg}"

            f.write(f"arguments = {cmd}\n")
            f.write(f"log = {log_file}\n")
            f.write(f"output = {out_file}\n")
            f.write(f"error = {err_file}\n")
            f.write("Queue\n\n\n")

    def submit(self, dag=False):
        """Submit the file to condor

        Parameters
        ----------
        dag : boolean
            whether to submit jobs using the dag script
        """

        file_path = os.path.join(self.folder_in_juseless, self.python_file)
        assert os.path.isfile(
            file_path
        ), "Your specified python file does not exist!"

        assert os.path.isdir(
            self.logs_dir
        ), "Your specified logs directory does not exist!"

        curr_dir = os.getcwd()
        os.chdir(self.folder_in_juseless)
        os.system(f"chmod +x {self.python_file}")
        if self.venv_activation_file is not None:
            os.system(f"chmod +x {self.venv_activation_file}")
        os.chdir(curr_dir)

        if not dag:
            os.system(f"condor_submit {self.submit_file}")
        elif dag:
            os.system(f"condor_submit_dag {self.dag}")

        self.submitted = True

    def write_job_submission_log(
        self, log_file=None, message="Jobs submitted!", file_type="txt"
    ):
        """A method to keep track of each time you submit jobs by logging time
        of submission, some other related information, and a user-defined
        submission message

        log_file : str
            name of file in which to log submission information
        message : str
            user-defined submission message to write to the log file
        file_type : str
            can be either "txt" or "csv" to write output to corresponding file
            format
        """

        header = False

        if log_file is None:
            log_file = self.submission_id

        log_file = f"{log_file}.{file_type}"

        now = datetime.now()
        d = now.strftime("%m/%d/%Y, %H:%M:%S")

        if os.path.isfile(log_file):
            file_mode = "a"
        else:
            file_mode = "w"
            header = True

        if file_type == "txt":
            with open(log_file, file_mode) as f:
                f.write(f"\n{d}\n\n")
                f.write(
                    f"Job is {self.cmd}!\n"
                    f"Submitted?: {self.submitted}\n"
                    f"Submission ID is {self.submission_id}\n"
                )
                f.write(f"{message}\n\n")

        if file_type == "csv":

            with open(log_file, file_mode) as f:
                if header:
                    f.write(
                        "date,time,command_line,"
                        "submission_id,message,submitted\n"
                    )
                    file_mode = "a"

            with open(log_file, file_mode) as f:
                f.write(
                    f"{d},{self.cmd},{self.submission_id},{message},"
                    f"{self.submitted}\n"
                )

    def write_venv_activation_file(self, which_venv=None, which_conda=None):
        """Writes bash script to initialise a python virtual environment

        Parameters:
        -----------
        which_venv : str or path
            directory to virtual env

        """

        assert (
            self.venv_activation_file is not None
        ), "You have not specified a venv activation file"

        activation_file = os.path.join(
            self.folder_in_juseless, self.venv_activation_file
        )

        if which_venv is not None:

            with open(activation_file, "w") as f:
                f.write("#!/usr/bin/bash\n")
                f.write(f"source {which_venv}/bin/activate\n")
                f.write("python3 $@\n")
                f.write("deactivate\n")

        else:
            raise ValueError(
                "Which virtual environment should be initialised?"
            )

        os.system(f"chmod +x {activation_file}")

    def add_sum_jobs(self, file_name, python_file, args=None):
        """Add a submit file for a python script that summarises jobs. Can be
        used in combination with a DAG script to automatically sum up results
        from individual jobs.

        Parameters
        ----------
        file_name : str
            name of file in which to write condor submit file
        python_file : str
            name of file to execute to sum up results of individial jobs
        args : list
            args to be passed over to the job which will summarise results
        """

        self.write_condor_settings(submit_file=file_name)
        self.add_job(args=args, submit_file=file_name, python_file=python_file)

        self.sum_submit = file_name

    def write_dag(self, dag_submit_fname):
        """Write a simple dag script to add a job that will summarise the
        results.

        Parameters
        ----------
        dag_submit_fname : str
            name of file to use for the dag script

        """

        dag = (
            f"JOB SINGLESUBJECT {self.submit_file}\n"
            f"JOB GROUP {self.sum_submit}\n"
            "PARENT SINGLESUBJECT CHILD GROUP\n"
        )

        with open(dag_submit_fname, "w") as submit_file:
            submit_file.write(dag)

        self.dag = dag_submit_fname

    def monitor_logs(self, submission_id=None, which=".log", display_time=5):
        if submission_id is None:
            submission_id = self.submission_id

        logs_path = os.path.join(
            os.path.abspath(self.folder_in_juseless), self.logs_dir
        )

        # list all available log files
        log_files = [
            f
            for f in os.listdir(logs_path)
            if os.path.isfile(os.path.join(logs_path, f))
        ]

        if which != "all":
            log_files = [f for f in log_files if which in f]

        log_files = [f for f in log_files if submission_id in f]

        for f in log_files:
            current_file = os.path.join(self.logs_dir, f)
            os.system(f"cat {current_file}")
            time.sleep(display_time)
