#!/usr/bin/env python3


import os
import sys
from itertools import product

import numpy as np

from etspredict.job_submission import HTCSubmissionTemplate

subjects = list(
    # np.loadtxt(os.path.join("..", "non_na_subjects.txt"), dtype=str)
    np.loadtxt(os.path.join("..", "remaining.txt"), dtype=str)
)


sessions = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]

atlases = {
    "schaefer400x17": (
        "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
    ),
    "schaefer300x17": (
        "Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
    ),
    "schaefer200x17": (
        "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
    ),
}


template = HTCSubmissionTemplate(
    "extract_timeseries.py",
    folder_in_juseless=os.path.join("..", "preprocessing_pipeline"),
    venv_activation_file="run_in_venv.sh",
    submit_file="submit_file.submit",
    n_cpus=1,
    requested_gb=12,
    requested_disk=20,
    logs_dir=os.path.join("..", "logs"),
)

template.write_condor_settings()
template.write_venv_activation_file(which_venv="~/.venvs/cleanenv")

i = 0
for subject in subjects:
    args = [subject]
    log_id = "_".join(args)
    for atlas, session in product(atlases, sessions):
        outpath = os.path.join(
            "/home",
            "lsasse",
            "remaining",
            subject,
            "GS",
            atlas,
            f"{subject}_{session}.npy",
        )
        if not os.path.isfile(outpath):
            template.add_job(args, job_specific_log=log_id)
            i += 1
            break
print(i)
# template.submit()

message = sys.argv[1]
template.write_job_submission_log(
    "connectome_extraction_jobs", message="check txt", file_type="csv"
)
template.write_job_submission_log(
    "connectome_extraction_jobs", message=message, file_type="txt"
)
