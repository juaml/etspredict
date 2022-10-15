import os
from itertools import product

from etspredict.job_submission import HTCSubmissionTemplate


def main():

    datasets = ["hcp", "hcp_aging"]
    sampling_strats = ["individual_bins", "combined_bins", "sequential"]
    preprocessing_levels = ["GS", "no_GS"]
    parcellations = ["schaefer200x17", "schaefer300x17", "schaefer400x17"]
    correlation_methods = ["pearson", "spearman"]
    path_to_venv = "~/.venvs/cleanenv"
    name_submit_file = (
        "submit_files_identification_full_fc/"
        "submit_file_identification_full_fc.submit"
    )
    template = HTCSubmissionTemplate(
        "identification_full_fc.py",
        folder_in_juseless=os.path.join("..", "exec"),
        venv_activation_file="run_in_venv.sh",
        submit_file=name_submit_file,
        n_cpus=1,
        requested_gb=25,
        logs_dir=os.path.join("..", "logs", "identification_full_fc"),
    )
    template.write_condor_settings()
    template.write_venv_activation_file(which_venv=path_to_venv)

    i = 0
    for (
        dataset,
        sampling_strat,
        preprocessing_level,
        parcellation,
        correlation_method,
    ) in product(
        datasets,
        sampling_strats,
        preprocessing_levels,
        parcellations,
        correlation_methods,
    ):

        if (dataset in ["hcp_aging"]) and (preprocessing_level in ["no_GS"]):
            continue

        args = [
            dataset,
            preprocessing_level,
            parcellation,
            "rss",
            correlation_method,
            sampling_strat,
        ]
        log_id = "_".join(args)
        template.add_job(args, job_specific_log=log_id)

        i += 1

    print(f"Submit file created for {name_submit_file} with {i} jobs!")


if __name__ == "__main__":
    main()
