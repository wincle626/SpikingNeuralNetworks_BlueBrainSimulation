#!/bin/bash
[ "$1" ] || (echo "Please enter name of executable" ; exit -1)

echo "running $@"
echo "job started on $(date)" > timing
time $@
echo "job finished on $(date)" >> timing
