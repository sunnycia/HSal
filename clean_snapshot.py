import shutil
import os, glob
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--snapshot_dir',type=str,default='')
parser.add_argument('--directory_list_file',type=str,default='')

args = parser.parse_args()

snapshot_dir = args.snapshot_dir
directory_list_file = args.directory_list_file
if directory_list_file != '':
    r_f = open(directory_list_file, 'r')
    lines = r_f.readlines()
    for line in lines:
        line = line.strip()
        if os.path.isdir(line):
            # print line, 'will be deleted.'
            shutil.rmtree(line)
            print line, 'is removed.'
    print "Done."
    exit()

delete_list = []
for root, dirs, files in os.walk(snapshot_dir, topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        ext = os.path.splitext(file_path)[-1]
        # print ext
        if ext=='.caffemodel' or ext=='.solverstate':
            print file_path
            delete_list.append(file_path)
# print 
if len(delete_list)==0:
    print 'No caffemodel or snapshot found, exit'
    exit()

confirm = raw_input("Warning: these %s file will be deleted, are you sure? (y/n)"%str(len(delete_list)))
if confirm=='y':
    # print '.'
    for file_path in delete_list:
        os.remove(file_path)
    print "Done."

else:
    print "Nothing done."
    exit()