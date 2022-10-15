#!/usr/bin/env python3

import os

import numpy as np

with open("submit_file.submit", "w") as f:

    settings = f"""
executable = /usr/bin/bash
transfer_executable = False
initial_dir= ../fs_stats
universe = vanilla
getenv = True
request_cpus = 1
request_memory = 12GB
request_disk = 5GB
"""
    f.write(settings)

subjects = np.loadtxt("subjects_hcp_aging.txt", dtype=str)

for subject in subjects:
    if not os.path.isdir(f"../fs_stats/subj_temp/{subject}_V1_MR/stats"):

        with open("submit_file.submit", "a") as f:
            q = f"""
arguments = ./get_subj_data.sh {subject}
log = ../logs/{subject}_$(Cluster).$(Process).log
output = ../logs/{subject}_$(Cluster).$(Process).out
error = ../logs/{subject}_$(Cluster).$(Process).err
Queue

"""

            f.write(q)

# template.submit()
