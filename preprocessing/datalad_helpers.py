import os
import tempfile
import time
import warnings
from os.path import expanduser

import datalad
import datalad.api as dl
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
home = expanduser("~")
get_exc = datalad.support.exceptions.IncompleteResultsError


class Fetcher:
    """A generic class to wrap up some common datalad functionalities
    This can be used to create custom wrappers for specific datasets like HCP
    or UKB.
    When implementing a wrapper for a specific dataset make sure to initialise
    the Fetcher class with sensible default values for that dataset in the init
    method and then make any required additions.
    For example, Fetcher classes will need a dataset name (initialise the
    self.dataset_name attribute). This is the name of the root folder of the
    dataset. For the HCP dataset the name is "HCP1200". This is
    important for the self.drop_and_remove() method to work correctly.
    Other attributes you may want to implement are subject, session, experiment
    etc. specific parameters. In the case of the HCP dataset there are
    attributes like self.EXPERIMENTS or self.RUNS.
    Attributes
    -----------
    self.directory : str or path
        absolute path leading up to the datalad dataset
    self.repo : str
        remote address for the datalad dataset to clone
    self.files : list of str or list of paths
        files the fetcher has fetched so far
        (relative paths from self.directory)
    """

    def __init__(
        self, repo=None, directory=None, log_filename="Fetcher_log.txt"
    ):
        """
        Parameters
        ----------
        repo : str
            link to remote of the datalad dataset
        directory : str or path
            directory in which to clone the dataset
        """

        self.directory = directory
        self.repo = repo
        self.log_filename = log_filename

    def log(self, log_str):
        """
        Write a string to the input log file
        Parameters:
        ---------
        log_filename : str or path
            name of the log file
        log_str : str
            string to be written in the log file
        Returns
        --------
            None
        """

        assert isinstance(
            log_str, str
        ), "Please provide input to the logging function as a string!"
        try:
            with open(self.log_filename, "a+") as f:
                f.write(log_str)
        except FileNotFoundError:
            print("Logfile does not exist!")
        finally:
            print(log_str)

    def clone_repo(self, repo=None, directory=None):
        """Clone a remote-hosted datalad dataset.
        The purpose of this method is to easily create a temp folder for the
        dataset to be installed if no other directory is specified, clone the
        repo and keep track of where the datalad dataset is installed by
        updating the self.directory attribute (if no specific directory is
        provided)
        Parameters
        ----------
        repo : str
            remote address from which clone. If not provided, fetcher will use
            the address at self.repo (from the init method). Raises an Error if
            no remote address has been provided during initialisation or use of
            the method.
        directory : str or path
            if datalad should not be cloned into self.directory then you can
            provide another directory here to specifiy where to install it
        """

        if self.repo is None:
            self.repo = repo
            if repo is None:
                raise ValueError(
                    "Which repo should be cloned? Please provide a link!"
                )

        if self.directory is None:

            self.log("No Directory was specified. Making tmp directory\n")
            self.directory = tempfile.mkdtemp()

        for i in range(50):
            try:
                dl.install(self.directory, source=self.repo)
                break
            except (get_exc):
                print(get_exc)
                print("Trying again in 60 seconds!")
                time.sleep(60)

        self.dataset = dl.Dataset(self.directory)

    def get_custom_file(self, path, tries=50, wait_period=1, inm7=False):
        """Get a file from a datalad dataset with an arbitrary path from that
        dataset. For more details on the datalad.api.get function check here:
        https://docs.datalad.org/en/stable/generated/datalad.api.get.html
        The purpose of this function is to easily get the file, and keep track
        of the files that have been downloaded so far, so that these can be
        easily removed later.
        Parameters
        -----------
        path : str or path
            path of file to get relative from self.directory (the directory
            in which the datalad dataset was installed)
        tries : int
            how many times fetcher should try to get a file
        wait_period : int
            how long to wait until trying again
        """

        if self.directory is None:
            raise ValueError(
                "Please clone the repository first. Then you can get files!"
            )

        self.log("---------------------------------------------------------\n")
        self.log(
            f"DATALAD GETTING \n{path} \nIN DIRECTORY: \n{self.directory}\n"
        )

        if inm7:
            self.dataset.get(path, source="inm7-storage", recursive=True)
        else:
            self.dataset.get(path)

        self.log("DATALAD DONE GETTING FILE\n")

        return os.path.join(self.dataset.pathobj, path)

    def drop_and_remove(self):
        """Drop all files and remove this dataset
        This method should be called at the end of every processing routine to
        clean up the temporary directory if keeping the dataset installed is
        undesired.
        """

        self.log("---------------------------------------------------------\n")
        self.log("Removing Dataset\n")
        self.log("---------------------------------------------------------\n")
        try:
            self.dataset.remove(recursive=True)
        except ValueError:
            os.chdir("..")
            self.dataset.remove(recursive=True)

    def __enter__(self):
        """Enter method for context manager"""
        self.clone_repo()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit method for context manager"""
        self.drop_and_remove()


class HCPFetcher(Fetcher):
    def __init__(
        self,
        repo=(
            "https://github.com/datalad-datasets/"
            "human-connectome-project-openaccess.git"
        ),
        *args,
        **kwargs,
    ):
        """Initialise HCPFetcher
        Parameters
        ----------
        repo : str
            link to remote of the datalad dataset
        directory : str or path
            directory in which to clone the dataset
        Attributes
        ----------
        self.EXPERIMENTS : dictionary
            dictionary with information about task fMRI sessions in the HCP
            dataset
        self.RUNS : list
            list of phase encoding directions in HCP datasets
        self.TR : float
            time resolution in seconds
        """

        Fetcher.__init__(self, repo=repo, *args, **kwargs)

        self.EXPERIMENTS = {
            "MOTOR": {"cond": ["lf", "rf", "lh", "rh", "t", "cue"]},
            "WM": {
                "cond": [
                    "0bk_body",
                    "0bk_faces",
                    "0bk_places",
                    "0bk_tools",
                    "2bk_body",
                    "2bk_faces",
                    "2bk_places",
                    "2bk_tools",
                ]
            },
            "EMOTION": {"cond": ["fear", "neut"]},
            "GAMBLING": {"cond": ["loss", "win"]},
            "LANGUAGE": {"cond": ["math", "story"]},
            "RELATIONAL": {"cond": ["match", "relation"]},
            "SOCIAL": {"cond": ["ment", "rnd"]},
        }
        self.RUNS = ["LR", "RL"]
        self.TR = 0.72

    def get_hcp_nifti(
        self,
        subject,
        session,
        phase_encoding,
        confounds_path=True,
        tries=50,
        wait_period=1,
    ):
        """
        This Method will use Datalad to get the HCP open access data for the
        specified subject and the specified session. In particular it gets the
        already preprocessed rest and task sessions as nifti files and return
        their paths.
        For more info on the Python API for Datalad check out:
        http://docs.datalad.org/en/stable/generated/datalad.api.Dataset.html
        Parameters
        ----------
        subject : str
            subject ID for subject to get
        session : str
            name of session to get. Options are "REST1", "REST2", "GAMBLING",
            "WM", "MOTOR", "LANGUAGE", "EMOTION", "RELATIONAL" and "SOCIAL"
        phase_encoding : str
            phase encoding direction to get. Options are "LR" or "RL"
        confounds_path : Bool
            whether to also return the path of the corresponding confounds file
            in the AML group directory on Juseless.
        tries : int
            How many times fetcher should try to get a file
        wait_period : int
            how long to wait between tries
        Returns
        -------
        nifti_path : str or path
            absolute path to nifti file fetched by datalad
        conf_path : str or path
            Path to corresponding confound file in AML group directory on
            Juseless. Only returned if 'confounds_path' is set to True.
        """

        conf_dir = "/data/group/appliedml/data/HCP1200_Confounds_tsv_withFD/"

        self.log(
            f"""
                PARAMETERS FOR FILE TO GET ARE:
                SUBJECT ID:                  {subject}
                SESSION                      {session}
                PHASE ENCODING DIRECTION     {phase_encoding}

            """
        )

        # use file parameters to get the corresponding nifti file from dataset
        # and the correct confounds file
        if session in ["REST1", "REST2"]:
            file_path = os.path.join(
                "HCP1200",
                f"{subject}",
                "MNINonLinear",
                "Results",
                f"rfMRI_{session}_{phase_encoding}",
                f"rfMRI_{session}_{phase_encoding}_hp2000_clean.nii.gz",
            )

            confounds_file = os.path.join(
                f"{subject}",
                "MNINonLinear",
                "Results",
                f"rfMRI_{session}_{phase_encoding}",
                f"Confounds_{subject}.tsv",
            )

        else:
            file_path = os.path.join(
                "HCP1200",
                f"{subject}",
                "MNINonLinear",
                "Results",
                f"tfMRI_{session}_{phase_encoding}",
                f"tfMRI_{session}_{phase_encoding}.nii.gz",
            )

            confounds_file = os.path.join(
                f"{subject}",
                "MNINonLinear",
                "Results",
                f"tfMRI_{session}_{phase_encoding}",
                f"Confounds_{subject}.tsv",
            )

        # if the nifti path is not correct datalad will throw an error
        nifti_path = os.path.join(self.directory, file_path)

        if confounds_path:
            conf_path = os.path.join(conf_dir, confounds_file)

            if not os.path.isfile(conf_path):
                warnings.warn(
                    f"The confound file does not exist in this path!"
                )

        self.get_custom_file(file_path, tries=tries, wait_period=wait_period)

        if confounds_path:
            return nifti_path, conf_path
        else:
            return nifti_path

    def load_evs(
        self, subject, session, phase_encoding, tries=50, wait_period=1
    ):
        """Load EVs (explanatory variables) data for one task experiment.
        Parameters
        ----------
        subject : str
            subject ID for subject to get
        session : str
            name of session to get. Options are "REST1", "REST2", "GAMBLING",
            "WM", "MOTOR", "LANGUAGE", "EMOTION", "RELATIONAL" and "SOCIAL"
        phase_encoding : str
            phase encoding direction to get. Options are "LR" or "RL"
        tries : int
            how many times datalad should try getting a file
        wait_period : int
            how long to wait in between tries
        Returns
        -------
        df_whole : pandas.DataFrame
            Dataframe with two columns: ["frame"] gives the indices of the
            corresponding frames in the nifti image of the task run.
            ["condition"] gives the experimental condition the subject was in
            during the corresponding frame.
        """

        frames_list = []
        task_key = f"tfMRI_{session}_{phase_encoding}"
        for cond in self.EXPERIMENTS[session]["cond"]:
            ev_file = (
                f"HCP1200/{subject}/MNINonLinear/Results/"
                f"{task_key}/EVs/{cond}.txt"
            )
            ev_file = self.get_custom_file(
                ev_file, tries=tries, wait_period=wait_period
            )
            ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
            ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
            # Determine when trial starts, rounded down
            start = np.floor(ev["onset"] / self.TR).astype(int)
            # Use trial duration to determine how many frames to include for
            # trial
            duration = np.ceil(ev["duration"] / self.TR).astype(int)
            # Take the range of frames that correspond to this specific trial
            frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
            frames = np.concatenate(frames)
            framewise_cond = []
            for i, frame in enumerate(frames):
                framewise_cond.append(cond)

            results = {"frame": frames, "condition": framewise_cond}
            dataframe = pd.DataFrame(results)
            frames_list.append(dataframe)

        df_whole = (
            pd.concat(frames_list)
            .sort_values(by="frame")
            .reset_index(drop=True)
        )

        return df_whole


# --------------------------------------------------------------------------- #


class HCPAging(Fetcher):
    def __init__(
        self,
        repo=(
            "https://github.com/datalad-datasets/"
            "human-connectome-project-openaccess.git"
        ),
        *args,
        **kwargs,
    ):
        """Initialise HCPAging
        Parameters
        ----------
        repo : str
            link to remote of the datalad dataset
        directory : str or path
            directory in which to clone the dataset
        """

        Fetcher.__init__(self, repo=repo, *args, **kwargs)

    def get_nifti(
        self, subject, session, phase_encoding, tries=50, wait_period=1
    ):

        path = os.path.join(
            "original",
            "hcp",
            "hcp_aging",
            f"{subject}_V1_MR",
            "MNINonLinear",
            "Results",
            f"rfMRI_{session}_{phase_encoding}",
            f"rfMRI_{session}_{phase_encoding}_hp0_clean.nii.gz",
        )
        """
        self.get_custom_file(
            path, tries=tries, wait_period=wait_period, inm7=True
        )
        """
        hcp_a_confounds = os.path.join(
            "/data",
            "project",
            "identification_prediction",
            "data",
            "HCP_A_Confounds",
            subject,
            f"sub-{subject}_task-rfMRI{session}"
            f"{phase_encoding}_desc-confounds_timeseries.tsv",
        )

        return path, hcp_a_confounds
