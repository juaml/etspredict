#!/usr/bin/env python3

import os
import random
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
from nilearn import image, maskers, masking

sys.path.append(os.path.join("..", ".."))
from datalad_helpers import HCPFetcher

#############################################################################
# Inputs and preprocessing parameters
#############################################################################


subject = sys.argv[1]
GS = True

sessions = ["REST1", "REST2"]
phase_encodings = ["LR", "RL"]

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


out_path = os.path.join("/home", "lsasse", "remaining", subject)
# out_path = os.path.join(
#    "..", "..", "..", "etspredict", "data", "hcp_ya", "timeseries", subject
# )

if GS:
    out_path = os.path.join(out_path, "GS")
else:
    out_path = os.path.join(out_path, "no_GS")

if not os.path.isdir(out_path):
    os.makedirs(out_path)


#############################################################################
# Get Data and Preprocess
#############################################################################


time.sleep(random.randint(0, 500))

with HCPFetcher() as f:

    for session, phase_encoding in product(sessions, phase_encodings):

        # get data from datalad
        nifti_path, conf_path = f.get_hcp_nifti(
            subject=subject, session=session, phase_encoding=phase_encoding
        )

        # select confounds
        confounds_df = pd.read_csv(conf_path, sep="\t")

        if GS:
            confounds_to_select = ["WM", "CSF", "GS"]

        else:
            confounds_to_select = ["WM", "CSF"]

        # prepare spike regressor
        fd = confounds_df["FD"]
        fd.loc[fd > 0.25] = 0
        fd.loc[fd != 0] = 1

        confounds_selected = []
        for column in confounds_df:
            for term in confounds_to_select:
                if term in column:
                    confounds_selected.append(column)

        confounds_filtered = [x for x in confounds_selected if "PCA" not in x]

        confounds_df = confounds_df[confounds_filtered]

        # calculate temporal derivatives
        additional_computes = pd.DataFrame()
        non_squares_columns = [
            x for x in confounds_df.columns if "^2" not in x
        ]
        non_squares = confounds_df[non_squares_columns]
        for name, column in non_squares.iteritems():
            additional_computes[f"{name}_derivative"] = np.append(
                np.diff(column), 0
            )

        for name, column in additional_computes.iteritems():
            additional_computes[f"{name}_squared"] = column**2

        confounds_df[additional_computes.columns] = additional_computes

        # add the spike regressor
        confounds_df["spike_regressor"] = fd

        if GS:
            assert confounds_df.shape == (1200, 13)
        else:
            assert confounds_df.shape == (1200, 9)

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

            print(atlas_name)
            outfile_path = os.path.join(out_path, atlas_name)
            if not os.path.isdir(outfile_path):
                os.makedirs(outfile_path)

            outfile = os.path.join(
                outfile_path, f"{subject}_{session}_{phase_encoding}.npy"
            )
            if os.path.isfile(outfile):
                continue

            parcellation = image.load_img(atlas_path)

            nifti_masker = maskers.NiftiLabelsMasker(parcellation)

            parcellated = nifti_masker.fit_transform(clean_nifti)
            np.save(outfile, parcellated)
