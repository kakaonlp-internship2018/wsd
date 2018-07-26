"""
transform.py
__author__ = 'jeff.yu (jeff.yu@kakaocorp.com)'
__copyright__ = 'No copyright, just copyleft!'
"""
import re
import os
import sys

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
REMOVE_SENSE = False

try:
    FR = open(sys.argv[1], 'r')
    FW = open(sys.argv[2], 'w')
except BaseException:
    print('usage: python3 transform.py source_file_name dist_file_name')
    sys.exit(1)

COUNT = 0
while True:
    LINE = FR.readLINE()
    if not LINE:
        break
    if COUNT % 3 == 1:
        LINE = LINE.replace("\n", "")
        if REMOVE_SENSE:
            LINE = re.sub(r'__[\d][\d]', "", LINE)
        LINE = re.split('[+ ]', LINE)
        FW.write(' '.join(LINE)+"\n")
    COUNT = COUNT+1

FR.close()
FW.close()
