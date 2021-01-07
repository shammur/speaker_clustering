# Installation

1- Create new invironment from `environment.yml`

`conda env create -f environment.yml`

2- Change the path to the spk clustering model `/home/stageapp/data/qats1/scripts/speaker_clustering/saver/` in `src/extract_embedding_from_model.py` script. 

3- Change the line `source ~/anaconda3/bin/activate ~/anaconda3/envs/inaseg` in `scripts/run_speaker_clustering.sh` to your anaconda directory.

# Testing

In order to test the clustering run `bash scripts/run_speaker_clustering.sh sample_data`



