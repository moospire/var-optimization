import os
import sys

file_dir = os.path.dirname(__file__)
# add below when breaking out into another file
if not file_dir in sys.path:
    sys.path.append(file_dir)
    print('adding {} to sys path'.format(file_dir))
