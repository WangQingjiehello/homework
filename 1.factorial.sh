#!/bin/bash

if [ "$1" = "" ]
then
	read -p "usage:1.factorial [n]"
	read -p "calculate a number's factorial"
	exit 
fi

fac()
{
	if [ $1 -le 0 ]
	then
		num=1
	else
		fac $(($1-1))
		num=$(($1*$num))
	fi
	return $num
}
fac $1
echo "$num"
