#!/bin/bash
if [ "$1" == "1" ] && [ "$2" != "" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python linear_reg.py $2 $3 $4
elif [ "$1" == "2" ] && [ "$2"!="" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python local_linear.py $2 $3 $4
elif [ "$1" == "3" ] && [ "$2" != "" ] && [ "$3" != "" ]; then
	python logistic_reg.py $2 $3
elif [ "$1" == "4" ] && [ "$2" != "" ] && [ "$3" != "" ] && [ "$4" != "" ]; then
	python gda.py $2 $3 $4
else
	echo "Not enough requirements"
	exit
fi