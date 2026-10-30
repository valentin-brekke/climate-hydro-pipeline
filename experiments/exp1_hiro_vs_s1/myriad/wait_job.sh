#!/bin/bash -l
# Usage: wait_job.sh JOB_ID MAX_SECONDS [running|done]
# Waits on Myriad (so the Mac side is a single ssh call) until the job is
# running ("running", the default) or has left the queue ("done"), or
# MAX_SECONDS pass. Then prints qstat and the tail of the job's stdout.
job=$1; max=$2; until=${3:-running}
for ((t = 0; t < max; t += 20)); do
    state=$(qstat | awk -v j="$job" '$1 == j {print $5}')
    [[ $until == running && -n $state && $state != qw ]] && break
    [[ $until == done && -z $state ]] && break
    sleep 20
done
qstat
log=$(ls ~/Scratch/climate-hydro/logs/*.o"$job" 2>/dev/null | head -1)
[[ -n $log ]] && { echo "--- $log"; tail -n 25 "$log"; }
