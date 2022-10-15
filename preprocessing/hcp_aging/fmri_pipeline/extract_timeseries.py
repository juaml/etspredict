#!/usr/bin/env python3

import os
import random
import sys
import tempfile
import time
from itertools import product

import numpy as np
import pandas as pd
from nilearn import image, maskers, masking

sys.path.append(os.path.join("..", ".."))
from datalad_helpers import HCPAging

#############################################################################
# Inputs and preprocessing parameters
#############################################################################


subject = sys.argv[1]
GS = True

sessions = ["REST1", "REST2"]
phase_encodings = ["AP", "PA"]

preprocessing_parameters = {
    "sessions": None,
    "detrend": True,
    "high_pass": 0.008,
    "low_pass": 0.08,
    "standardize": False,
}


############################################################################
# Parcellations
############################################################################


parc_dir = os.path.join("..", "..", "parcellations")
atlases = {
    "schaefer400x17": os.path.join(
        parc_dir,
        "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
    ),
    "schaefer300x17": os.path.join(
        parc_dir,
        "Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
    ),
    "schaefer200x17": os.path.join(
        parc_dir,
        "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
    ),
}


#############################################################################
# Prepare path for output
#############################################################################


out_path = os.path.join(
    "..", "..", "..", "etspredict", "data", "hcp_aging", "timeseries", subject
)

if GS:
    out_path = os.path.join(out_path, "GS")
else:
    out_path = os.path.join(out_path, "no_GS")

if not os.path.isdir(out_path):
    os.makedirs(out_path)

time.sleep(random.randint(0, 300))
with HCPAging() as f:

    for session, phase_encoding in product(sessions, phase_encodings):

        # get data from datalad
        nifti_path, conf_path = f.get_nifti(
            subject=subject, session=session, phase_encoding=phase_encoding
        )

        tmp = tempfile.mkdtemp()

        time.sleep(random.randint(0, 300))
        os.system(f"./datalad_get_subj.sh {tmp} {nifti_path}")

        nifti_path = os.path.join(tmp, "inm7_ds", nifti_path)
        assert os.path.isfile(nifti_path), f"{nifti_path} \n" "IS NOT A FILE!"

        # select confounds
        confounds_df = pd.read_csv(conf_path, sep="\t")
        bad_tp = confounds_df["badTP"]

        if GS:
            confounds_to_select = ["WM", "CSF", "GS"]

        else:
            confounds_to_select = ["WM", "CSF"]

        confounds_df = confounds_df[confounds_to_select]

        # calculate temporal derivatives
        for name, column in confounds_df.iteritems():
            confounds_df[f"{name}_derivative"] = np.append(np.diff(column), 0)

        for name, column in confounds_df.iteritems():
            confounds_df[f"{name}_squared"] = column**2

        # add the spike regressor
        confounds_df["spike_regressor"] = bad_tp

        if GS:
            assert confounds_df.shape == (478, 13)
        else:
            assert confounds_df.shape == (478, 9)

        confounds = confounds_df.values

        nifti_image = image.load_img(nifti_path)
        preprocessing_parameters["t_r"] = nifti_image.header.get_zooms()[3]

        mask_img = masking.compute_brain_mask(
            nifti_image, mask_type="whole-brain"
        )
        clean_nifti = image.clean_img(
            imgs=nifti_image,
            confounds=confounds,
            mask_img=mask_img,
            **preprocessing_parameters,
        )

        for atlas_name, atlas_path in atlases.items():

            outfile_path = os.path.join(out_path, atlas_name)
            if not os.path.isdir(outfile_path):
                os.makedirs(outfile_path)

            outfile = os.path.join(
                outfile_path, f"{subject}_{session}_{phase_encoding}.npy"
            )

            parcellation = image.load_img(atlas_path)

            nifti_masker = maskers.NiftiLabelsMasker(parcellation)

            parcellated = nifti_masker.fit_transform(clean_nifti)
            print(parcellated)
            np.save(outfile, parcellated)
