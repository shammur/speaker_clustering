#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Shammur A Chowdhury
#
# Copyright (C) 2020, Qatar Computing Research Institute, HBKU, Qatar
# Created on : Mon March  30
# Last Update: Tue March  31


import argparse
import os
import warnings
import distutils.util
import time
import pandas as pd
import ray
from inaSpeechSegmenter import Segmenter, seg2csv
from pydub import AudioSegment

ray.init()

@ray.remote(
    memory=4000 * 1024 * 1024,
)
def _do_segmentation_(input_files, chunk_id, seg, odir) :

    start_time = time.time()
    odircsv=os.path.join(odir,"seg_csv")
    odirw = os.path.join(odir, "seg_wav")
    segment_dict = {}

    for i, e in enumerate(input_files):
        print('processing file %d-%d/%d: %s' % (chunk_id, i + 1, len(input_files), e))
        e=e.rstrip().lstrip()

        base, _ = os.path.splitext(os.path.basename(e))
        seg2csv(seg(e), '%s/%s.csv' % (odircsv, base))
        segs=_do_wav_split_(e,os.path.join(odircsv,base+".csv"),odirw)
        segment_dict[base]=segs

    elapsed_time = time.time() - start_time
    print(format(elapsed_time) + ' seconds elapsed for ' + str(chunk_id))
    return segment_dict

def _do_wav_split_(win, csvin, woutdir) :
    base,_= os.path.splitext(os.path.basename(win))
    segs=[]

    audio = AudioSegment.from_wav(win)
    data=pd.read_csv(csvin, sep='\t')
    for index, rows in data.iterrows():
        lab=rows['labels']
        if 'speech' in lab:
            start = float(rows['start'])*1000 #in milliseconds
            end = float(rows['stop'])*1000 #in milliseconds
            s='{:010.5f}'.format(start)
            e = '{:010.5f}'.format(end)
            fsegment=base+'_'+(s.replace('.',''))+'-'+(e.replace('.',''))
            segs.append(fsegment+' '+ base+' '+str(float(rows['start'])) + ' '+ str(float(rows['stop'])))
            file_name=os.path.join(woutdir,  fsegment+'.wav')
            segment = audio[start:end]
            segment.export(file_name, format="wav")  # Exports to a wav file in the current path.
    return segs






# Configure command line parsing
parser = argparse.ArgumentParser(description='Do Speech/Music and Male/Female segmentation. Store segmentations into CSV files')
parser.add_argument('-i', '--input',  help='Input media to analyse. '
                                           'Provide a list of full paths (/home/david/test.mp3 /tmp/mymedia.avi), '
                                           'or a regex input pattern ("/home/david/myaudiobooks/*.mp3")',
                                            required=True)
parser.add_argument('-o', '--output_directory', help='Directory used to store segmentations. '
                                                     'Resulting segmentations have same base name as the corresponding input media, '
                                                     'with csv extension. Ex: mymedia.MPG will result in seg_csv/mymedia.csv', required=True)

parser.add_argument('-d', '--vad_engine', choices=['sm', 'smn'], default='smn', help="Voice activity detection (VAD) engine to be used (default: 'smn'). "
                                                "'smn' split signal into 'speech', 'music' and 'noise' (better). "
                                                "'sm' split signal into 'speech' and 'music' and do not take noise into account, which is either classified as music or speech.")
parser.add_argument('-g', '--detect_gender', choices = ['true', 'false'], default='True', help="(default: 'true'). "
                                                "If set to 'true', segments detected as speech will be splitted into 'male' and 'female' segments. "
                                                "If set to 'false', segments corresponding to speech will be labelled as 'speech' (faster)")

parser.add_argument('-p', '--split', default=1, help="(default: '1'). "
                                                "Number of jobs to run in parallel.")



args = parser.parse_args()

## Check out directory exists and has permission to write
odir = args.output_directory

if not os.path.exists(odir):
    os.makedirs(odir)
if not os.path.exists(odir+"/seg_wav"):
    os.makedirs(odir+"/seg_wav")
if not os.path.exists(odir+"/seg_csv"):
    os.makedirs(odir + "/seg_csv")
assert os.access(odir, os.W_OK), 'Directory %s is not writable!' % odir

# Do processings


"""
Input to the Segmenter:
'vad_engine' can be 'sm' (speech/music) or 'smn' (speech/music/noise)
        'sm' was used in the results presented in ICASSP 2017 paper
                and in MIREX 2018 challenge submission
        'smn' has been implemented more recently and has not been evaluated in papers

'detect_gender': if False, speech excerpts are return labelled as 'speech'
        if True, speech excerpts are splitted into 'male' and 'female' segments
"""

detect_gender = bool(distutils.util.strtobool(args.detect_gender))

# load neural network into memory, may last few seconds
seg = Segmenter(vad_engine=args.vad_engine, detect_gender=detect_gender)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


input_files = []

with open(args.input,'r') as file:
    data=file.read().splitlines()
    for f in data:
        input_files.append(f)

print('Processing total files %d:' % (len(input_files)))
no_of_sentences=len(input_files)
num_jobs=int(args.split)

chunk_size=int(no_of_sentences/num_jobs)

chunks = [input_files[x:x+chunk_size] for x in range(0, len(input_files), chunk_size)]
ret_ids=[]
for i, e in enumerate(chunks):
    ret_ids.append(_do_segmentation_.remote(e,i,seg,odir))


sement_chunks= (ray.get(ret_ids))
print('done segmentation ......')
fout=open(os.path.join(odir,'segments'), 'w')

for seg_chunk in sement_chunks:
    for key in seg_chunk:
        seg_lst=seg_chunk[key]
        for entry in seg_lst:
            fout.write(entry+"\n")

fout.close()
print('done creating segment file ......')





