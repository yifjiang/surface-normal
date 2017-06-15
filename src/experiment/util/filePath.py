import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-From', default='train_NYU_subset_NC_normal_75684_all.csv')
parser.add_argument('-length', default=4,type=int)
parser.add_argument('-to', default='jiang_train_NYU_subset_NC_normal_75684_all.csv')
args = parser.parse_args()

f=open(args.From,'r')
w=open(args.to,'w')
length = args.length
for l in f:
	parts = l.split(',')
	temp = parts[0].split('/')[-length:]
	substitute = ['../..']
	substitute.extend(temp)
	newPath = ['/'.join(substitute)]
	newPath.extend(parts[1:])
	newLine = ','.join(newPath)
	w.write(newLine)

f.close()
w.close()