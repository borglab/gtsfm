#!/bin/bash

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE

pytest tests --cov gtsfm
coverage report

##########################################################
# Test with flake8
##########################################################

pip install flake8
flake8 --max-line-length 120 --ignore E201,E202,E203,E231,W291,W293,E303,W391,E402,W503 gtsfm