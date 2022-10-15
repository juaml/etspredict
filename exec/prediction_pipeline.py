import os

import numpy as np
import pkg_resources
from scipy.stats import zscore
from sklearn.model_selection import GroupKFold, RepeatedKFold

from etspredict.data.hcp_ya_pheno.get_data import load_behavioural_data
from etspredict.pipelines.ioutils import configure_output_name
from etspredict.prediction.model_grids import get_model_grid
from etspredict.prediction.pipe import Pipeline
from etspredict.prepare_data import prepare_connectomes as pc
from etspredict.prepare_data import prepare_phenotypes as pp


def load_subject_list(dataset):

    path = os.path.join("..", "etspredict", "data", f"{dataset}_subjects.txt")
    stream = pkg_resources.resource_stream(__name__, path)
    return list(np.loadtxt(stream, dtype=str))


def main(
    dataset,
    preprocessing_level,
    rank,
    target,
    sampling_strat,
    threshold,
    parcellation,
    model_name,
    sc_free,
):

    print("Starting prediction pipeline!")
    # initialise the correct confounds
    confounds = pp.prepare_confound_names(dataset, target)

    # initialise output directory
    if model_name in ["cbpm"]:
        output_dir = os.path.join(
            "..",
            "results",
            "cbpm",
            configure_output_name(target, sampling_strat, sc_free),
            dataset,
            preprocessing_level,
        )
        preprocess_X = model_name
        model_name = "lin"

    else:
        output_dir = os.path.join(
            "..",
            "results",
            configure_output_name(target, sampling_strat, sc_free),
            dataset,
            preprocessing_level,
        )
        if target not in ["sex", "Gender"]:
            preprocess_X = None

    pipe_name = (
        f"amplranking-{rank}_parcellation-{parcellation}_target-{target}_"
        f"session-allsessions_model-{model_name}_"
        f"preproclevel-{preprocessing_level}zscored_threshold-{threshold}"
    )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # initialise correct sessions
    sessions_dict = {
        "hcp": ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"],
        "hcp_aging": ["REST1_AP", "REST1_PA", "REST2_AP", "REST2_PA"],
    }
    sessions = sessions_dict[dataset]

    # subjects
    subjects = load_subject_list(dataset)

    # initialise sampling function
    sampling_dict = {
        "individual_bins": ("all_slices", f"Rank {rank}"),
        "combined_bins": ("combined_bins", rank),  # I know
        "sequential": ("high_low", rank),
    }
    if rank in ["full"]:
        func = "high_low"
        rank_name = ("high",)
        threshold = 100
    else:
        func, rank_name = sampling_dict[sampling_strat]

    # load connectomes
    print("Loading connectomes, this will take a while ...")
    df_timeframe_connectomes = pc.load_group_connectomes(
        dataset=dataset,
        subjects=subjects,
        preprocessing_level=preprocessing_level,
        sessions=sessions,
        function=func,
        parcellation=parcellation,
        rank=rank_name,
        criterion="rss",
        threshold=threshold,
    )
    df_timeframe_connectomes.columns = df_timeframe_connectomes.columns.map(
        str
    )
    features = list(df_timeframe_connectomes.columns)

    if sc_free:
        df_timeframe_connectomes = pc.regress_out_sc_from_dataframe(
            df_timeframe_connectomes
        )
        subjects = list(df_timeframe_connectomes.index)
        sc_subs_file = "../etspredict/data/hcp_subjects_sc_fc.txt"
        if not os.path.isfile(sc_subs_file):
            np.savetxt(sc_subs_file, subjects, fmt="%s")

    # zscore connectomes for pearson kernel
    if model_name in ["kernelridge_scikit"]:
        df_timeframe_connectomes = df_timeframe_connectomes.apply(
            lambda V: zscore(V), axis=1
        )

    # target data and confounds to add
    target_data, confound_data = pp.load_target_confounds(
        subjects, dataset, target, confounds
    )

    confounds = list(confound_data.columns)
    df_timeframe_connectomes[target] = target_data
    df_timeframe_connectomes[confounds] = confound_data
    # at this point prediction data is ready for julearn

    # distinguish between regression and classification
    if target in ["sex", "Gender"]:
        problem_type = "binary_classification"
        preprocess_X = "remove_confound"
        preprocess_y = None
        scoring = [
            "balanced_accuracy",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
        ]
    else:
        problem_type = "regression"

        preprocess_y = "remove_confound"
        scoring = [
            "neg_mean_absolute_error",
            "corr",
            "neg_root_mean_squared_error",
            "r2",
        ]

    # models and model parameters
    model, m_params = get_model_grid(
        model_name=model_name, problem_type=problem_type
    )
    if target not in ["sex", "Gender"]:
        m_params["remove_confound"] = "passthrough"

    # prepare cv
    if dataset == "hcp":
        # get family information for group cv
        groups = "Family_ID"
        df_timeframe_connectomes[groups] = load_behavioural_data(groups)
        cv = GroupKFold(n_splits=10)
        m_params["search_params"] = dict(cv=GroupKFold(n_splits=5))
    else:
        groups = None
        cv = RepeatedKFold(n_splits=5, n_repeats=5)

    pipeline = Pipeline(
        prediction_data=df_timeframe_connectomes,
        model=model,
        model_params=m_params,
        output_directory=output_dir,
        pipeline_name=pipe_name,
        features=features,
        target=target,
        confounds=confounds,
        problem_type=problem_type,
        preprocess_X=preprocess_X,
        preprocess_y=preprocess_y,
        return_estimator="all",
        scoring=scoring,
        seed=1234567,
        cv=cv,
        groups=groups,
    )
    pipeline.run_julearn()
    pipeline.save_scores()


if __name__ == "__main__":
    import sys

    main(
        dataset=sys.argv[1],
        preprocessing_level=sys.argv[2],
        rank=sys.argv[3],
        target=sys.argv[4],
        sampling_strat=sys.argv[5],
        threshold=float(sys.argv[6]),
        parcellation=sys.argv[7],
        model_name=sys.argv[8],
        sc_free=bool(int(sys.argv[9])),
    )
