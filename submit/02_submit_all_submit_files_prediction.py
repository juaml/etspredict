#!/usr/bin/env python3

import os

submit_files = os.listdir("submit_files")

for f in submit_files:
    os.system(f"condor_submit submit_files/{f}")
