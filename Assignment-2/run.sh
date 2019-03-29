#!/bin/bash
if [ "$1" == "1" ] && [ "$2" != "" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python naive_bayes.py $2 $3 $4
elif [ "$1" == "2" ] && [ "$2"!="" ] && [ "$3" != "" ] && [ "$4" != "" ] && [ "$5" != "" ]; then
	python svm.py $2 $3 $4 $5
else
	echo "Supply appropriate and correct arguments"
	exit
fi