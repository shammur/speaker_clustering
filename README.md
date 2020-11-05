# speaker_clustering

run bash script in scripts/run_speaker_clustering.sh
with stage 2 and 3
stage=1 splits the data into subtask - no need now.
To run the 2nd and 3rd stage you will need sample_data/wav.scp, sample_data/segments and segmented_wav

First step extract embedding #stage2
Then do a speaker clustering #stage=3

Keep the parameters same in the bash script except the TOTAL_SPLIT=2 #1 if no split is done

The output of /src/spk_clustering.py is seg2spk where the last column represent the speaker id. 
--max_spks 5 #represent the maximum possible speaker - you can use 10
