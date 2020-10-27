import os, sys
import time
import librosa
import argparse



def calculate_wav_duration(filename):
    return librosa.get_duration(filename=filename)

def get_duration(filelist):
    input_files = []
    files2duration = {}
    with open(filelist, 'r') as file:
        data = file.read().splitlines()
        for f in data:
            input_files.append(f)





parser = argparse.ArgumentParser(description="Generate Duration Statistics from Wav List", add_help=True)
parser.add_argument("--wavlist", type=str, help="List of full paths of wav data",required=True)
parser.add_argument("--outfile", type=str, help="csv file with the duration info")

args = parser.parse_known_args()[0]

segments = open(SOURCE_FOLDER+'/segments').readlines()
