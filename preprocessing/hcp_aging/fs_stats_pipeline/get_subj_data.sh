#!/usr/bin/bash -x
############################################################################
############################################################################


subj=${1}_V1_MR


############################################################################
############################################################################

CWD=$(pwd)
output=subj_temp
datalad_temp=$(mktemp -d)


############################################################################
############################################################################


cd ${datalad_temp}
datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git ${datalad_temp}

cd original

echo $(pwd)

datalad get -n .
cd hcp
datalad get -n .
cd hcp_aging
datalad get -n .
cd ${subj}
datalad get -n .
echo $(pwd)

cd T1w
echo $(pwd)
datalad get -n .
datalad get ${subj}/stats -s inm7-storage

cp -L -r ${subj} ${CWD}/subj_temp/.

############################################################################
############################################################################


cd ${CWD}
rm -rf datalad_temp
