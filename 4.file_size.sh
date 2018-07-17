#!/bin/bash
if [ ! -n "$2" -o ! -n "$4" ]
then
	echo "usage: 4.file_size.sh [-n N] [-d DIR]"
	echo "Show top N largest files/directories"
else
cd "$4"
echo "The largest files/dirctories in "$4" are:"
du -sh * | sort -rh | head -$2 | cat -n
fi
