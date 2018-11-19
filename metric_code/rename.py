import sys
import os

if not len(sys.argv)==2:
    print "one argument required."
    exit()

imageDir = sys.argv[1]
# print imageDir
imageList = os.listdir(imageDir)
# print imageList
print "Renaming files in",imageDir,"...",
for filename in imageList:
    oldPath = os.path.join(imageDir, filename)
    filename = filename.replace('_HDR', '')
    newPath = os.path.join(imageDir, filename)
    # print oldPath, newPath ;exit()
    os.rename(oldPath, newPath)
print "Done."

