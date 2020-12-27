# Installation

1- Create new invironment from `environment.yml`

`conda env create -f environment.yml`

2- Change the path to the model `/home/stageapp/data/qats1/scripts/speaker_clustering/saver/` in `src/extract_embedding_from_model.py`. 

3- Change the following in `utt2spk.sh`

- the path to the installed env `source /home/stageapp/data/qats1/espnet/tools/venv/bin/activate /home/stageapp/data/qats1/espnet/tools/venv/envs/inaseg`
- the path to where the script extracted `script_path=/home/stageapp/data/qats1/scripts/speaker_clustering`

