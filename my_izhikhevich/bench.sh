#!/bin/bash
# runs the two versions of izhikevich (parallel/non-parallel)
# saves network size/sim time in two separate files
# 
# use compare_bench.m to plot differences

SAVEFILE_OMP="bench-omp.txt"
SAVEFILE_NONOMP="bench-nonomp.txt"
MAXRUNS=10

CMD_NONOMP="./myIzhik-nonomp"
CMD_OMP="./myIzhik-omp"
STDARGS="-f /dev/null"

if [ -f $SAVEFILE_OMP ] ; then 
	echo "file $SAVEFILE_OMP exists" 
	exit -1 
fi
if [ -f $SAVEFILE_NONOMP ] ; then 
	echo "file $SAVEFILE_NONOMP exists" 
	exit -1
fi

for base in $(seq 100 50 1000) ; do 
	inh=$(($base+100))
	exc=$(($inh*4))
	tot=$(($exc+$inh))
	for runs in $(seq 1 $MAXRUNS) ; do
		randseed=$RANDOM
		ARGS="$STDARGS -e $exc -i $inh -s $randseed"

		echo "run $runs of $MAXRUNS, running non-OMP with $tot neurones, random seed $randseed"
		t=$($CMD_NONOMP $ARGS | grep "Elapsed tim" | awk ' { print $4 } ')
		echo "$tot $t" >> $SAVEFILE_NONOMP

		echo "run $runs of $MAXRUNS, running OMP with $tot neurones"
		t=$($CMD_OMP $ARGS | grep "Elapsed tim" | awk ' { print $4 } ')
		echo "$tot $t" >> $SAVEFILE_OMP
		#read 
	done
done
