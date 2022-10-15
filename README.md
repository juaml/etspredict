# etspredict
Using edge time series to estimate FC at different levels of co-fluctuation for individual-level predictions

# Set up environment

This project uses julearn for machine-learning and cross-validation. It is a 
library built on top of scikit-learn, and built with specific neuroscientific
challenges such as cross-validation consistent deconfounding in mind.
Check out its GitHub and documentation here:

* Github: https://github.com/juaml/julearn
* Documentation: https://juaml.github.io/julearn/main/index.html

The specific commit hash used is this one:
* 6c94a2d3682799e74e99db184c713797b7094d22

Thus to use the same version, you should probably install from GitHub using
this commit hash.

Make a virtual environment:

```sh
python3 -m venv /path/to/newvenv
source /path/to/newvenv/bin/activate

pip install -U pip

```

Then go to the location at which you want to install julearn:

```sh
git clone https://github.com/juaml/julearn.git
cd julearn
git checkout 6c94a2d3682799e74e99db184c713797b7094d22
pip install .
```

Afterwards you can install etspredict. Again, go to the location at which
you want to install it. Then:

```sh

git clone https://github.com/juaml/etspredict.git
cd etspredict
pip insall -r requirements.txt
pip install .
```

In order to actually run code, you will need to obtain access to data from 
the human connectome project neuroimaging data and behavioural data 
(https://github.com/datalad-datasets/human-connectome-project-openaccess; https://db.humanconnectome.org/)

Denoised time series should be placed at etspredict/etspredict/data/hcp_ya 
and etspredict/etspredict/data/hcp_aging for the hcp young adult and aging
datasets respectively.

