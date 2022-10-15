



temp_dir=${1}
nifti_path=${2}
cw_dir=$(pwd)

cat << print_args

DATALAD IS GETTING 

${nifti_path}

FROM

${temp_dir}

print_args

cd ${temp_dir}

datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git inm7_ds
cd inm7_ds
datalad get -s inm7-storage ${nifti_path}

cd ${cw_dir}
