#!/usr/bin/env python3

import os
import sys
from itertools import product

import numpy as np

from etspredict.job_submission import HTCSubmissionTemplate

subjects = list(np.loadtxt("subjects_hcp_aging.txt", dtype=str))

missing_subs = list(np.loadtxt("missing_subs.txt", dtype=str))

subjects = [x for x in subjects if x not in missing_subs]

sessions = ["REST1_AP", "REST1_PA", "REST2_AP", "REST2_PA"]

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
    folder_in_juseless=os.path.join("..", "fmri_pipeline"),
    venv_activation_file="run_in_venv.sh",
    submit_file="submit_file.submit",
    n_cpus=1,
    requested_gb=12,
    # requested_disk=20,
    logs_dir=os.path.join("..", "logs"),
)

template.write_condor_settings()
template.write_venv_activation_file(which_venv="~/.venvs/myvenv")

for subject in subjects:
    args = [subject]
    log_id = "_".join(args)
    add = []
    for atlas, session in product(atlases, sessions):
        outpath = os.path.join(
            "..",
            "timeseries",
            subject,
            "GS",
            atlas,
            f"{subject}_{session}.npy",
        )
        if not os.path.isfile(outpath):
            add.append("YES")

        if add:
            template.add_job(args, job_specific_log=log_id)


# template.submit()

message = sys.argv[1]
template.write_job_submission_log(
    "connectome_extraction_jobs", message="check txt", file_type="csv"
)
template.write_job_submission_log(
    "connectome_extraction_jobs", message=message, file_type="txt"
)
