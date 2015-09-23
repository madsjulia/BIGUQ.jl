#include("testoed.jl")
import testoed
import BlackBoxOptim
#import ProfileView

nummodelruns = 100000
hakunamatata = 1.
numlikelihoods = 60
numhorizons = 301
numobsrealizations = 1000
acceptableprobabilityoffailure = 0.1
#paramsmin, paramsmax, bigoed = testoed.makebigoed1()
@everywhere srand(0)
modelparams = BlackBoxOptim.Utils.latin_hypercube_sampling(testoed.paramsmin, testoed.paramsmax, nummodelruns)
@everywhere srand(0)
residualdistribution = testoed.makeresidualdistribution(testoed.residualdistributionparamsmin(0.), testoed.bigoed1.obslocations, testoed.bigoed1.obstimes, testoed.bigoed1.obsmodelindices, [], [], [])
for decisionparam in testoed.decisionparams
	llhoods = Float64[]
	maxllhood = -Inf
	for i = 1:nummodelruns
		push!(llhoods, Distributions.logpdf(residualdistribution, testoed.bigoed1.obs - testoed.bigoed1.models[1](modelparams[:, i], decisionparam, testoed.bigoed1.obslocations, testoed.bigoed1.obstimes)))
		maxllhood = max(maxllhood, llhoods[end])
	end
	println(exp(sort(llhoods)[end-20:end] - sort(llhoods)[end]))
	for i = 1:nummodelruns
		println(decisionparam, " ", testoed.gethorizonoffailure(modelparams[:, i], decisionparam), " ", exp(-maxllhood + Distributions.logpdf(residualdistribution, testoed.bigoed1.obs - testoed.bigoed1.models[1](modelparams[:, i], decisionparam, testoed.bigoed1.obslocations, testoed.bigoed1.obstimes))))
	end
end
#=
bigdts = BIGUQ.makebigdts(testoed.bigoed1)
maxfailureprobs, horizons, badlikelihoodparams = BIGUQ.getrobustnesscurve(bigdts[1], hakunamatata, numlikelihoods; numhorizons = 11)
=#
@time decisionprobabilities = BIGUQ.dobigoed(testoed.bigoed1, hakunamatata, numlikelihoods, numhorizons, numobsrealizations, acceptableprobabilityoffailure, modelparams)
println(decisionprobabilities)
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
