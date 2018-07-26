"""
transform.py
__author__ = 'jeff.yu (jeff.yu@kakaocorp.com)'
__copyright__ = 'No copyright, just copyleft!'
"""
import re
import sys

def main():
    """
    this is main function
    """
    remove_sense = False

    try:
        fr_source = open(sys.argv[1], 'r')
        fw_dist = open(sys.argv[2], 'w')
    except BaseException:
        print('usage: python3 transform.py source_file_name dist_file_name')
        sys.exit(1)

    count = 0
    while True:
        line = fr_source.readline()
        if not line:
            break
        if count % 3 == 1:
            line = line.replace("\n", "")
            if remove_sense:
                line = re.sub(r'__[\d][\d]', "", line)
            line = re.split('[+ ]', line)
            fw_dist.write(' '.join(line)+"\n")
        count = count+1

    fr_source.close()
    fw_dist.close()

if __name__ == '__main__':
    main()
