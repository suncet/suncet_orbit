#!/bin/zsh

brew install gcc
brew install cmake

current_env=$(basename $CONDA_PREFIX)
cd "${CONDA_PREFIX}/lib/python3.10/site-packages/"
git clone https://github.com/space-physics/msise00
pip install -e msise00

# MSISE won't work until the source code has been built in python; only needs to be done once
ipython <<EOF
import msise00.base
msise00.base.build()
EOF
