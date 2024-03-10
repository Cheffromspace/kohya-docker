#!/usr/bin/env bash
echo 'Updating kohya'
cd /workspace/kohya_ss
source venv/bin/activate
git checkout master
git pull
pip3 install -r requirements.txt
pip3 install .
echo 'kohya updated'
