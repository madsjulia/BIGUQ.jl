#include("testoed.jl")
import testoed
import BlackBoxOptim
#import ProfileView

@everywhere srand(0)
nummodelruns = 25
hakunamatata = 1.
numlikelihoods = 60
numhorizons = 101
numobsrealizations = 100
acceptableprobabilityoffailure = 0.1
#paramsmin, paramsmax, bigoed = testoed.makebigoed1()
modelparams = BlackBoxOptim.Utils.latin_hypercube_sampling(testoed.paramsmin, testoed.paramsmax, nummodelruns)
for decisionparam in testoed.decisionparams
	for i = 1:nummodelruns
		println(decisionparam, " ", testoed.gethorizonoffailure(modelparams[:, i], decisionparam))
	end
end
#=
Profile.init(10 ^ 7, 0.005)
Profile.clear()
decisionprobabilities = @profile BIGUQ.dobigoed(bigoed, hakunamatata, numlikelihoods, numhorizons, numobsrealizations, acceptableprobabilityoffailure, modelparams)
ProfileView.svgwrite("bigoed.svg")
=#
#=
@time decisionprobabilities = BIGUQ.dobigoed(testoed.bigoed1, hakunamatata, numlikelihoods, numhorizons, numobsrealizations, acceptableprobabilityoffailure, modelparams)
println(decisionprobabilities)
=#
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
