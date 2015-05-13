import testoed
@everywhere srand(0)
bigoed = testoed.makebigoed1()
#=
bigdts = BIGUQ.makebigdts(bigoed)
decisionindex = BIGUQ.makedecision(bigdts, 0.1, 1., 19, 11; robustnesspenalty=bigoed.robustnesspenalty)
println(decisionindex)
=#
bigdts2 = BIGUQ.makebigdts(bigoed, 1, 0 * ones(4))
decisionindex2 = BIGUQ.makedecision(bigdts2, 0.1, 1., 19, 11; robustnesspenalty=bigoed.robustnesspenalty)
println(decisionindex2)
#=
for bigdt in bigdts
	@time maxfailureprobs, horizons, badlikelihoodparams = BIGUQ.getrobustnesscurve(bigdt, 1., 19; numhorizons = 11)
	BIGUQ.printresults(maxfailureprobs, horizons, badlikelihoodparams)
end
=#
