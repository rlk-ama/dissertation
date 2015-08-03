number=$1
sum_optimal=0
sum_prior=0
dest="/home/raphael/profiling.txt"
for (( i=0; i < number; i++ ))
do
    echo "toto"
    kernprof -l "examples/ricker.py" --particles 5000
    file="/home/raphael/dissertation/ricker.py.lprof"
    python3 -m line_profiler $file > $dest
    t1=`awk '/proposal.sample/ {printf $4}' $dest`
    t2=`awk '/proposal.density/ {printf $4}' $dest`
    sum_optimal=`echo "$t1+$t2" | bc`
    kernprof -l "examples/ricker.py" --filter_proposal prior --particles 5000
    python3 -m line_profiler $file > $dest
    t1=`awk '/prior.sample/ {printf $4}' $dest`
    t2=`awk '/prior.density/ {printf $4}' $dest`
    sum_prior=`echo "$t1+$t2" | bc`
done
av_optimal=`echo "$sum_optimal/$number" | bc`
av_prior=`echo "$sum_prior/$number" | bc`
echo $av_optimal
echo $av_prior
