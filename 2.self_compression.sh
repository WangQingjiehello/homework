#!/bin/bash

file=$1
path=$2

if [ -z $file ]
then
	echo "usage: 2.self_compression.sh[--list]"
	echo "or"
	echo "[Source cmpressed file] [Destination path]"
	echo "Self compression according to the file name suffix"
fi

if [ "$file" = "--list" ]
then
      echo "Supported file types :zip tar tar.gz tar.bz2"
fi

if [ -n "$file" -a -n "$path" ]
then
	hz=${file##*.}

	case $hz in
	'zip')
	  eval "unzip $file -d $path";;
	'tar')
	eval "tar xvf $file -C $path";;
	'gz')
	eval "tar zxvf $file -C $path";;
	'bz2')
	eval "tar jxvf $file -C $path";;
	*)
	echo "error";;
	esac
fi 
