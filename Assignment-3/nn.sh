#!/bin/bash
if [ "$1" != "" ] && [ "$2" != "" ] && [ "$3" != "" ]; then
	python neural_nets.py $1 $2 $3
else
	echo "Supply appropriate and correct arguments"
	exit
fi