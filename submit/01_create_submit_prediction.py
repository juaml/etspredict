import os
from itertools import combinations, product

from etspredict.data.hcp_ya_pheno.get_data import load_info_targets
from etspredict.job_submission import HTCSubmissionTemplate


def create_submit(
    dataset,
    target_cat,
    sampling_strat,
    preprocessing_level,
    parcellation,
    path_to_venv,
    sc_free,
):

    if target_cat in ["sex"]:

        bins_dict = {
            "hcp": [str(i) for i in range(1, 21)],
            "hcp_aging": [str(i) for i in range(1, 9)],
        }

        model_names = ["rbf_svm", "linear_svm", "ridgeclassifier"]
        target_dict = {"hcp": ["Gender"], "hcp_aging": ["sex"]}

    elif target_cat in ["age"]:

        bins_dict = {
            "hcp": [str(i) for i in range(1, 21)],
            "hcp_aging": [str(i) for i in range(1, 9)],
        }

        model_names = ["kernelridge_scikit"]
        target_dict = {
            "hcp": ["Age_in_Yrs"],
            "hcp_aging": ["age"],
        }

    elif target_cat in ["psychometric"]:

        bins_dict = {
            "hcp": ["1", "5", "10", "15", "20"],
            "hcp_aging": [str(i) for i in range(1, 9)],
        }

        hcpya_targets = load_info_targets(
            category="Cognition", column="HCP field"
        )
        hcpya_targets = [x for x in hcpya_targets if "Comp" not in x]
        hcpya_targets += list(
            load_info_targets(
                category="In-Scanner Task Performance", column="HCP field"
            )
        )

        # personality
        hcpya_targets += list(
            load_info_targets(category="Personality", column="HCP field")
        )

        model_names = ["kernelridge_scikit", "cbpm"]
        target_dict = {
            "hcp": hcpya_targets,
            "hcp_aging": [
                "nih_dccs_unadjusted",
                "tpvt_uss",
                "nih_fluidcogcomp_unadjusted",
                "nih_crycogcomp_unadjusted",
            ],
        }

    targets = target_dict[dataset]
    bins = bins_dict[dataset]

    ranks_dict = {
        "individual_bins": bins,
        "combined_bins": ["_".join([x, y]) for x, y in combinations(bins, 2)],
        "sequential": ["high", "low"],
    }
    ranks = ranks_dict[sampling_strat]

    if dataset in ["hcp"]:
        threshol_dict = {
            "individual_bins": ["5"],
            "combined_bins": ["5"],
            "sequential": [
                "1",
                "2",
                "4",
                "5",
                "10",
                "15",
                "20",
                "25",
                "30",
                "40",
                "50",
                "100",
            ],
        }
    if dataset in ["hcp_aging"]:
        threshol_dict = {
            "individual_bins": ["12.5"],
            "combined_bins": ["12.5"],
            "sequential": [
                "1",
                "2",
                "4",
                "5",
                "10",
                "15",
                "20",
                "25",
                "30",
                "40",
                "50",
                "100",
            ],
        }
    thresholds = threshol_dict[sampling_strat]

    if bool(sc_free):
        name_submit_file = os.path.join(
            "submit_files_prediction",
            f"{dataset}_{target_cat}_{sampling_strat}_"
            f"{preprocessing_level}_{parcellation}_sc_free.submit",
        )
    else:
        name_submit_file = os.path.join(
            "submit_files_prediction",
            f"{dataset}_{target_cat}_{sampling_strat}_"
            f"{preprocessing_level}_{parcellation}.submit",
        )

    template = HTCSubmissionTemplate(
        "prediction_pipeline.py",
        folder_in_juseless=os.path.join("..", "exec"),
        venv_activation_file="run_in_venv.sh",
        submit_file=name_submit_file,
        n_cpus=1,
        requested_gb=12,
        logs_dir=os.path.join("..", "logs", "prediction"),
    )

    template.write_condor_settings()
    template.write_venv_activation_file(which_venv=path_to_venv)

    i = 0
    for rank, target, threshold, model_name in product(
        ranks, targets, thresholds, model_names
    ):

        args = [
            dataset,
            preprocessing_level,
            rank,
            target,
            sampling_strat,
            threshold,
            parcellation,
            model_name,
            str(sc_free),
        ]
        log_id = "_".join(args)
        template.add_job(args, job_specific_log=log_id)

        i += 1

    print(f"Submit file created for {name_submit_file} with {i} jobs!")

    return i


def main():

    datasets = ["hcp", "hcp_aging"]
    target_cats = ["age", "sex", "psychometric"]
    sampling_strats = ["individual_bins", "combined_bins", "sequential"]
    preprocessing_level = "GS"
    parcellation = "schaefer200x17"
    path_to_venv = "~/.venvs/cleanenv"
    sc_free_list = [0, 1]

    total_jobs = []
    j = 0
    for dataset, target_cat, sampling_strat, sc_free in product(
        datasets, target_cats, sampling_strats, sc_free_list
    ):
        if (bool(sc_free)) and (dataset in ["hcp_aging"]):
            continue

        n_jobs = create_submit(
            dataset,
            target_cat,
            sampling_strat,
            preprocessing_level,
            parcellation,
            path_to_venv,
            sc_free,
        )
        j += 1
        total_jobs.append(n_jobs)

    total_jobs = sum(total_jobs)
    print("--------------------------------------------------------------")
    print(f"{j} submit files created!")
    print(f"This results in {total_jobs} total jobs!")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    main()
