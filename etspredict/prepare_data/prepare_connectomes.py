import numpy as np
import pandas as pd
import statsmodels.api as sm

from etspredict.data import load_struct_connectomes
from etspredict.data.hcp_aging.load import HCPAgingTimeseries
from etspredict.data.hcp_ya.load import HCPTimeseries

from .. import selective_connectomes as sc


def load_timeseries(
    dataset, subject, parcellation, session, preprocessing=None
):
    """This function can be used to load timeseries that are preprocessed
    and pre-packaged for this project specifically.

    Parameters
    ----------

    dataset : str
        can be "hcp" for the HCP young adult dataset or "hcp_aging" for the
        HCP aging dataset
    subject : str
        Subject identifier
    parcellation : str
        can be name of any of the preprocessed parcellations, i.e.
        "schaefer200x17"
    session : str
        name of scanning session, i.e. "REST1_LR"
    preprocessing : str
        only refers to preprocessing with or without global signal regression
        ("GS" or "no_GS")

    Returns : np.array
    -------------------
        Timepoint x Parcel array of subject timeseries
    """

    # datasets to choose from
    datasets = {
        "hcp": HCPTimeseries,
        "hcp_aging": HCPAgingTimeseries,
    }

    # initialise a dataset object
    subject_data = datasets[dataset]()
    subject_data.load(
        subject=subject,
        session=session,
        parcellation=parcellation,
        preprocessing=preprocessing,
    )

    # hcp aging timeseries are cut in order to be divisible into 8 chunks
    # of 55 frames to enable easier comparison to HCP-YA
    if dataset == "hcp_aging":
        timeseries = subject_data.timeseries[20:460, :]
    else:
        timeseries = subject_data.timeseries

    return timeseries


def load_group_connectomes(
    dataset,
    subjects,
    sessions,
    preprocessing_level="None",
    parcellation="schaefer200x17",
    function="high_low",
    rank="high",
    criterion="rss",
    threshold=5,
):
    """Loads timeseries from one of the pre-packaged datasets and constructs
    connectomes for each subject from the list according to one of the
    functions and input parameters from the etspredict library (i.e. different
    sampling strategies of time frames).

    Parameters
    -----------
    dataset : str
        "hcp" or "hcp_aging"
    subjects : str or list of str
        str if one subject otherwise list of subject identifiers to load
    sessions : str or list of str
        str if one session otherwise list of session identifiers across which
        participant data can be averaged
    preprocessing_level : str
        refers to global signal regression
    parcellation : str
        can be name of any of the preprocessed parcellations, i.e.
        "schaefer200x17"
    function : str
        predefined name of functions available to construct connectomes
        according to different sampling strategies
    rank : str or list of str
        rank of frames to be used along the specified criterion, i.e. rank
        according to RSS magnitude. Depends on selected function
    threshold : int
        percentage of frames to be used in connectome construction

    Returns
    --------
    connectomes : pd.DataFrame or dict
        if one rank is specified it returns a pd.DataFrame where each row
        corresponds to a subject connectome. If a list of ranks is specified,
        it returns a dict with ranks as keys and corresponding connectome
        dataframes as values

    """

    if isinstance(subjects, str):
        subjects = [subjects]
    elif not isinstance(subjects, list):
        print("subjects should be list or string!")

    parameters = {
        "criterion": criterion,
        "threshold": threshold,
    }

    functions = {
        "all_slices": sc.select_all_slice_connectomes,
        "high_low": sc.select_high_and_low_connectome,
        "combined_bins": sc.all_combined_bins_connectomes,
    }

    assert (
        function in functions.keys()
    ), f"function must be a string: {functions.keys()}"

    executable_func = functions[function]

    if isinstance(rank, str):
        gc_list = []
        for subject in subjects:
            session_data_list = []
            for session in sessions:
                timeseries = load_timeseries(
                    dataset,
                    subject,
                    parcellation,
                    session,
                    preprocessing=preprocessing_level,
                )

                result = executable_func(timeseries, **parameters)
                session_connectome = result[rank]
                session_data_list.append(session_connectome)
            gc_list.append(pd.concat(session_data_list, axis=1).mean(axis=1))

        df_group_connectomes = pd.concat(gc_list, axis=1)
        df_group_connectomes = df_group_connectomes.T
        df_group_connectomes.index = subjects
        df_group_connectomes.columns = df_group_connectomes.columns.map(str)

        return df_group_connectomes

    elif isinstance(rank, list):
        rank_names = []
        gc_dict = {}
        n_subs = len(subjects)
        n_edges = None
        n_ranks = None
        for i_sub, subject in enumerate(subjects):
            print(subject)
            session_data_list = []
            for session in sessions:
                timeseries = load_timeseries(
                    dataset,
                    subject,
                    parcellation,
                    session,
                    preprocessing=preprocessing_level,
                )
                session_data_list.append(
                    executable_func(timeseries, **parameters)
                )
            result = pd.concat(session_data_list).groupby(level=0).mean()
            if not rank_names:
                rank_names = list(result.columns)
                rank_inds = [
                    i for i, val in enumerate(rank_names) if val in rank
                ]
                n_edges, n_ranks = result.shape
                result_3d = np.zeros((n_edges, n_ranks, n_subs))

            result_3d[:, :, i_sub] = result.values

        for i, r in zip(rank_inds, rank):
            rank_df = pd.DataFrame(result_3d[:, i, :].T)
            rank_df.columns.map(str)
            rank_df.index = subjects
            gc_dict[r] = rank_df

        return gc_dict


def remove_sc_from_fc(sc, fc):

    sc_copy = np.array(sc.copy())
    fc_copy = np.array(fc.copy())
    sc_copy = sm.add_constant(sc_copy)

    model = sm.OLS(fc_copy, sc_copy)
    results = model.fit()

    return results.resid


def regress_out_sc_from_dataframe(FC_df, parcellation="schaefer200x17"):

    df_sc = load_struct_connectomes(parcellation=parcellation)
    subs_to_exclude = [x for x in FC_df.index if x not in df_sc.index]
    FC_df = FC_df.drop(index=subs_to_exclude)
    fc_residuals_list = []
    # take functional connectomes and remove info from structural connectomes
    for subject, fc in FC_df.iterrows():
        sc_sub = df_sc.loc[subject]
        fc_residuals_list.append(pd.Series(remove_sc_from_fc(sc_sub, fc)))

    df_fc_residuals = pd.concat(fc_residuals_list, axis=1).T
    df_fc_residuals.index = FC_df.index
    df_fc_residuals.columns = FC_df.columns
    return df_fc_residuals
