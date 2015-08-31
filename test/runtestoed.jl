import testoed
import BlackBoxOptim
import ProfileView

@everywhere srand(0)
nummodelruns = 500
hakunamatata = 1.
numlikelihoods = 25
numhorizons = 101
numobsrealizations = 30
acceptableprobabilityoffailure = 0.1
paramsmin, paramsmax, bigoed = testoed.makebigoed1()
modelparams = BlackBoxOptim.Utils.latin_hypercube_sampling(paramsmin, paramsmax, nummodelruns)
Profile.init(10 ^ 7, 0.005)
Profile.clear()
decisionprobabilities = @profile BIGUQ.dobigoed(bigoed, hakunamatata, numlikelihoods, numhorizons, numobsrealizations, acceptableprobabilityoffailure, modelparams)
ProfileView.svgwrite("bigoed.svg")
#decisionprobabilities = BIGUQ.dobigoed(bigoed, hakunamatata, numlikelihoods, numhorizons, numobsrealizations, acceptableprobabilityoffailure, modelparams)
#println(decisionprobabilities)
#=
bigdts = BIGUQ.makebigdts(bigoed)
decisionindex = BIGUQ.makedecision(bigdts, 0.1, 1., 19, 11; robustnesspenalty=bigoed.robustnesspenalty)
println(decisionindex)
=#
#=
bigdts2 = BIGUQ.makebigdts(bigoed, 1, 0 * ones(4))
decisionindex2 = BIGUQ.makedecision(bigdts2, 0.1, 1., 19, 11; robustnesspenalty=bigoed.robustnesspenalty)
println(decisionindex2)
=#
#=
for bigdt in bigdts
	@time maxfailureprobs, horizons, badlikelihoodparams = BIGUQ.getrobustnesscurve(bigdt, 1., 19; numhorizons = 11)
	BIGUQ.printresults(maxfailureprobs, horizons, badlikelihoodparams)
end
=#
