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
        with open(sys.argv[1], 'r') as fr_source, open(sys.argv[2], 'w') as fw_dist:
            for count, line in enumerate(fr_source):
                if count % 3 == 1:
                    line = line.rstrip('\r\n')
                    if remove_sense:
                        line = re.sub(r'__[\d][\d]', "", line)
                    line = re.split('[+ ]', line)
                    print(' '.join(line), file=fw_dist)

    except BaseException:
        print('usage: python3 transform.py source_file_name dist_file_name')
        sys.exit(1)


if __name__ == '__main__':
    main()
