import os
from itertools import product

import numpy as np
import pkg_resources

from etspredict.job_submission import HTCSubmissionTemplate


def load_subject_list(dataset):

    path = os.path.join("..", "etspredict", "data", f"{dataset}_subjects.txt")
    stream = pkg_resources.resource_stream(__name__, path)

    return list(np.loadtxt(stream, dtype=str))


def main():

    subjects = load_subject_list("hcp")
    sessions = [
        "_".join([a, b]) for a, b in product(["REST1", "REST2"], ["LR", "RL"])
    ]

    parcellations = ["schaefer200x17", "schaefer300x17", "schaefer400x17"]
    correlation_methods = ["pearson", "spearman"]
    path_to_venv = "~/.venvs/cleanenv"
    name_submit_file = (
        "submit_files_correlation_rss_fd/submit_file_correlation_rss_fd.submit"
    )

    template = HTCSubmissionTemplate(
        "rss_fd.py",
        folder_in_juseless=os.path.join("..", "exec"),
        venv_activation_file="run_in_venv.sh",
        submit_file=name_submit_file,
        n_cpus=1,
        requested_gb=12,
        logs_dir=os.path.join("..", "logs", "rss_fd_correlation"),
    )
    template.write_condor_settings()
    template.write_venv_activation_file(which_venv=path_to_venv)

    i = 0
    for subject, session, parcellation, correlation_method in product(
        subjects, sessions, parcellations, correlation_methods
    ):

        args = [subject, session, parcellation, correlation_method]
        if not os.path.isfile(
            f"../results/gs_rss_correlations/"
            f"{subject}_{session}_{parcellation}_GS_{correlation_method}.txt"
        ) or not os.path.isfile(
            f"../results/gs_rss_correlations/"
            f"{subject}_{session}_{parcellation}"
            f"_no_GS_{correlation_method}.txt"
        ):

            log_id = "_".join(args)
            template.add_job(args, job_specific_log=log_id)
            i += 1

    print(f"Submit file created for {name_submit_file} with {i} jobs!")


if __name__ == "__main__":
    main()
