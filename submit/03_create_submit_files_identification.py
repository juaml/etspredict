import os
from itertools import product

from etspredict.job_submission import HTCSubmissionTemplate


def main():

    datasets = ["hcp"]
    sampling_strats = ["individual_bins"]
    preprocessing_levels = ["GS"]
    parcellations = ["schaefer200x17"]
    correlation_methods = ["pearson"]
    path_to_venv = "~/.venvs/cleanenv"
    name_submit_file = (
        "submit_files_identification/submit_file_identification.submit"
    )

    template = HTCSubmissionTemplate(
        "identification_pipeline.py",
        folder_in_juseless=os.path.join("..", "exec"),
        venv_activation_file="run_in_venv.sh",
        submit_file=name_submit_file,
        n_cpus=1,
        requested_gb=25,
        logs_dir=os.path.join("..", "logs", "identification"),
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
