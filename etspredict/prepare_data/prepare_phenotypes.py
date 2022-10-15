import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ..data.hcp_aging_pheno.load import load_hcp_aging_pheno
from ..data.hcp_ya_pheno.get_data import load_behavioural_data, load_fd_data


def prepare_confound_names(dataset, target):
    """Name of confounds selected as a function of dataset and prediction
    target.
    """

    if target in ["sex", "Gender"]:
        confounds_dict = {
            "hcp": ["Age_in_Yrs", "FS_BrainSeg_Vol", "SSAGA_Educ", "FD"],
            "hcp_aging": ["BrainSegVol", "age", "FD", "bkgrnd_education"],
        }
    elif target in ["Age_in_Yrs", "age"]:
        confounds_dict = {"hcp": ["Gender", "FD"], "hcp_aging": ["sex", "FD"]}
    else:
        confounds_dict = {
            "hcp": ["Gender", "Age_in_Yrs", "FD"],
            "hcp_aging": ["age", "sex", "FD"],
        }

    return confounds_dict[dataset]


def load_target_confounds(subjects, dataset, target, confounds):
    """Prepare targets and confounds for a dataset specific
    prediction pipeline.

    Parameters
    -----------
    subjects : list of str
        subject identifiers
    dataset: str
        "hcp" or "hcp_aging"
    target : str
        field name of target to load
    confounds : str or list of strings
        field names of confounds to load

    Returns
    --------
    target_data, confound_data : tuple of pd.DataFrames

    """
    confound_data = pd.DataFrame()

    if isinstance(confounds, str):
        confounds = [confounds]

    if dataset == "hcp":
        target_data = load_behavioural_data(
            datafields=target, subjects=subjects
        )

        confound_data = load_behavioural_data(
            datafields=confounds, subjects=subjects
        )

        if "FD" in confounds:
            confound_data["FD"] = load_fd_data(
                subjects=subjects, sessions=["REST1", "REST2"]
            ).mean(axis=1)
    if dataset == "hcp_aging":

        # files that contain behavioural variables of interest in HCP
        # Aging datasaet
        hcp_aging_files = [
            "hcp_aging_cogcomp01.txt",
            "hcp_aging_dccs01.txt",
            "hcp_aging_tpvt01.txt",
        ]

        # I select data for all targets I want to predict
        targets = [
            "nih_dccs_unadjusted",
            "tpvt_uss",
            "nih_fluidcogcomp_unadjusted",
            "nih_crycogcomp_unadjusted",
        ]

        # target can be one of the above or age and sex but nothing else
        if target not in ["age", "sex"]:
            assert (
                target in targets
            ), f"Target name incorrect, target can be {targets}"

        # load all files that may be relevant, these can be concatenated
        # using subject id as identifier/index
        hcp_aging_df_list = []
        for f in hcp_aging_files:
            df = load_hcp_aging_pheno(f)
            # drop descriptive row
            df = df.drop(labels=0)
            df = df.reset_index(drop=True)
            df = df.set_index("src_subject_id")
            df.index = df.index.astype(str)
            hcp_aging_df_list.append(df)

        all_data = pd.concat(hcp_aging_df_list, axis=1)
        target_data = all_data[targets]
        sel_subj_targ_data = target_data.loc[subjects]
        sel_subj = sel_subj_targ_data.dropna()
        # age is given in months
        # assert sel_subj.shape == (558, 4), (
        #    "Number of subjects not as expected, or number of potential"
        #    " targets not as expected"
        # )

        # from the concatenated data we can extract confounds and targets
        # and split them up

        # confound data
        data_for_confounds = df.loc[sel_subj.index]
        confound_data["age"] = (
            data_for_confounds["interview_age"].astype(int) / 12
        )
        confound_data["sex"] = data_for_confounds["sex"]

        # Load confounds based on freesurfer statistics if necessary:
        fs_stats = load_hcp_aging_pheno("HCP_AGING_FS_STATS.txt")
        fs_stats_index = [x[:10] for x in fs_stats["Measure:volume"]]
        fs_stats.index = fs_stats_index
        fs_stats = fs_stats.loc[confound_data.index]

        # load ssaga info for education
        ssaga = load_hcp_aging_pheno("HCP_AGING_SSAGA.txt")
        ssaga = ssaga.drop(labels=0)
        ssaga = ssaga.set_index("src_subject_id")
        ssaga = ssaga.loc[confound_data.index]
        ssaga = ssaga.drop(columns=("sex"))

        # FD
        FD_data = load_hcp_aging_pheno("HCP_AGING_FD_VALUES.csv")
        FD_data = FD_data.set_index("subject")
        FD_data = FD_data.loc[sel_subj.index]
        confound_data = pd.concat(
            [fs_stats, FD_data, confound_data, ssaga], axis=1
        )

        # if age and sex are targets, then info for these must be extracted
        # from confound dataset
        if target not in ["age", "sex"]:
            target_data = sel_subj[target]
            target_data = target_data.astype(float)
        else:
            target_data = confound_data[target]

        confound_data = confound_data[confounds]

    confound_data = confound_data.dropna()

    # in a last step, all data that consists of strings (i.e. categorial)
    # is OneHotEncoded

    cat_confounds = pd.DataFrame()
    confound_data_copy = confound_data.copy()
    dtypes = confound_data_copy.dtypes.to_dict()
    for col_name, typ in dtypes.items():
        ohe_enc = OneHotEncoder(drop="first", sparse=False)
        if typ == "O":
            ohe_confound = pd.DataFrame(
                ohe_enc.fit_transform(
                    np.array(confound_data_copy[col_name]).reshape(-1, 1)
                )
            )
            for i, col in enumerate(ohe_confound):
                cat_confounds[f"{col_name}_{i}"] = ohe_confound[col]

            confound_data_copy = confound_data_copy.drop(columns=col_name)

    cat_confounds.index = confound_data_copy.index
    confound_data_copy[cat_confounds.columns] = cat_confounds

    if target in ["sex", "Gender"]:
        ord_enc = OrdinalEncoder()
        target_data = ord_enc.fit_transform(target_data.values.reshape(-1, 1))

    return target_data, confound_data_copy
