for i in {1..250}
do
	./waf --run "scratch/MLASce/MLASce --RunNum=$(($i))"
done