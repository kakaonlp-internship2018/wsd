import re
import os
import sys

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
REMOVE_SENSE = False

try:
    fr = open(sys.argv[1], 'r')
    fw = open(sys.argv[2], 'w')
except:
    print('usage: python3 transform.py source_file_name dist_file_name')
    sys.exit(1)

count = 0
while True:
    line = fr.readline()
    if not line: break
    if count%3 == 1:
        line = line.replace("\n","")
        if REMOVE_SENSE : line = re.sub('__[\d][\d]',"",line)
        line = re.split('[+ ]',line)
        fw.write(' '.join(line)+"\n")
    count = count+1

fr.close()
fw.close()

