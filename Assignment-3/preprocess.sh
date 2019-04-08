#!/bin/bash
if [ "$1" != "" ] && [ "$2" != "" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python pre_process.py $1 $2 $3 $4
else
	echo "Supply appropriate and correct arguments"
	exit
fi