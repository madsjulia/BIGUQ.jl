type BigOED
	models::Array{Function, 1}#an array of functions that takes a vector of uncertain parameters, a vector of (certain) decision parameters, and two arrays: one of spatial coordinates, and another of times. It returns an array containing the results.
	#the different models represent different things that could be measured at different times/locations
	obs::Vector
	obslocations::Vector
	obstimes::Vector
	obsmodelindices::Array{Int64, 1}#an index that indicates which model would be used to represent the measurement found in the corresponding location of the obs array
	proposedlocations::Array{Array{Array{Float64, 1}, 1}, 1}
	proposedtimes::Array{Array{Float64, 1}, 1}
	proposedmodelindices::Array{Array{Int64, 1}, 1}#say which model the proposed data collection corresponds to
	makeresidualdistribution::Function#takes the params from the info-gap model as well as the obslocations, obstimes, proposedlocations[i], proposedtimes[i]
	residualdistributionparamsmin::Function
	residualdistributionparamsmax::Function
	nominalparams::Vector#nominal parameters for the model
	performancegoalsatisfied::Function#a function that take parameters, decision parameters, and a horizon of uncertainty
	logprior::Function
	decisionparams::Array{Array{Float64, 1}, 1}#an array of decision parameter arrays representing different possible decisions
	robustnesspenalty::Array{Float64, 1}#an array indicating how much robustness this decision costs
end

#makes the bigdts for each possible decision assuming that no more observations will be made
function makebigdts(bigoed::BigOED)
	function makeloglikelihood(likelihoodparams::Vector, decisionindex::Int64)
		const constlikelihoodparams = copy(likelihoodparams)
		const proposedlocations = []
		const proposedtimes = []
		const proposedmodelindices = []
		const residualdistribution = bigoed.makeresidualdistribution(constlikelihoodparams, bigoed.obslocations, bigoed.obstimes, bigoed.obsmodelindices, proposedlocations, proposedtimes, proposedmodelindices)
		function loglikelihood(params::Vector)
			results = Array(Float64, length(bigoed.obs))
			for i = 1:length(bigoed.models)
				goodindices = (bigoed.obsmodelindices .== i)
				results[goodindices] = bigoed.models[i](params, bigoed.decisionparams[decisionindex], bigoed.obslocations[goodindices], bigoed.obstimes[goodindices])
			end
			residuals = bigoed.obs - results
			retval = Distributions.logpdf(residualdistribution, residuals)
			return retval
		end
		return loglikelihood
	end
	bigdts = Array(BigDT, length(bigoed.decisionparams))
	for i = 1:length(bigoed.decisionparams)
		bigdts[i] = BIGUQ.BigDT(likelihoodparams->makeloglikelihood(likelihoodparams, i), bigoed.logprior, bigoed.nominalparams, bigoed.residualdistributionparamsmin, bigoed.residualdistributionparamsmax, (params::Array{Float64, 1}, horizon::Float64)->bigoed.performancegoalsatisfied(params, bigoed.decisionparams[i], horizon))
	end
	return bigdts
end

#make bigdts for each possible decision assuming that the proposedobs are observed
function makebigdts(bigoed::BigOED, proposedindex, proposedobs)
	function makeloglikelihood(likelihoodparams::Vector, decisionindex::Int64)
		const constlikelihoodparams = copy(likelihoodparams)
		const proposedlocations = bigoed.proposedlocations[proposedindex]
		const proposedtimes = bigoed.proposedtimes[proposedindex]
		const proposedmodelindices = bigoed.proposedmodelindices[proposedindex]
		const residualdistribution = bigoed.makeresidualdistribution(constlikelihoodparams, bigoed.obslocations, bigoed.obstimes, bigoed.obsmodelindices, proposedlocations, proposedtimes, proposedmodelindices)
		function loglikelihood(params::Vector)
			results = Array(Float64, length(bigoed.obs) + length(proposedtimes))
			for i = 1:length(bigoed.models)
				goodindicesshort = (bigoed.obsmodelindices .== i)
				goodindiceslong = [goodindicesshort; fill(false, length(proposedmodelindices))]
				results[goodindiceslong] = bigoed.models[i](params, bigoed.decisionparams[decisionindex], bigoed.obslocations[goodindicesshort], bigoed.obstimes[goodindicesshort])
			end
			for i = 1:length(bigoed.models)
				goodindicesshort = (proposedmodelindices .== i)
				goodindiceslong = [fill(false, length(bigoed.obsmodelindices)); goodindicesshort]
				results[goodindiceslong] = bigoed.models[i](params, bigoed.decisionparams[decisionindex], proposedlocations[goodindicesshort], proposedtimes[goodindicesshort])
			end
			residuals = [bigoed.obs; proposedobs] - results
			retval = Distributions.logpdf(residualdistribution, residuals)
			return retval
		end
		return loglikelihood
	end
	bigdts = Array(BigDT, length(bigoed.decisionparams))
	for i = 1:length(bigoed.decisionparams)
		bigdts[i] = BIGUQ.BigDT(likelihoodparams->makeloglikelihood(likelihoodparams, i), bigoed.logprior, bigoed.nominalparams, bigoed.residualdistributionparamsmin, bigoed.residualdistributionparamsmax, (params::Array{Float64, 1}, horizon::Float64)->bigoed.performancegoalsatisfied(params, bigoed.decisionparams[i], horizon))
	end
	return bigdts
end

function generateproposedobs(bigoed::BigOED, proposedindex, numobsrealizations; thinning=100, burnin=int(1e4))
	#setup and do the mcmc sampling
	likelihoodparams = bigoed.residualdistributionparamsmin(0.)
	residualdistribution = bigoed.makeresidualdistribution(likelihoodparams, bigoed.obslocations, bigoed.obstimes, bigoed.obsmodelindices, [], [], [])
	function loglikelihood(params::Vector)
		results = Array(Float64, length(bigoed.obs))
		for i = 1:length(bigoed.models)
			goodindices = (bigoed.obsmodelindices .== i)
			results[goodindices] = bigoed.models[i](params, bigoed.decisionparams[1], bigoed.obslocations[goodindices], bigoed.obstimes[goodindices])
		end
		residuals = bigoed.obs - results
		retval = Distributions.logpdf(residualdistribution, residuals)
		return retval
	end
	mcmcmodel = Lora.model(loglikelihood, init=bigoed.nominalparams)
	rmw = Lora.RAM(1e-1, 0.3)
	smc = Lora.SerialMC(nsteps=thinning * numobsrealizations + burnin, burnin=burnin, thinning=thinning)
	mcmcchain = Lora.run(mcmcmodel, rmw, smc)
	#use the mcmc samples to generate realizations of the proposed obs
	proposedlocations = bigoed.proposedlocations[proposedindex]
	proposedtimes = bigoed.proposedtimes[proposedindex]
	proposedmodelindices = bigoed.proposedmodelindices[proposedindex]
	proposedobsarray = Array(Array{Float64, 1}, numobsrealizations)
	for i = 1:numobsrealizations
		proposedobsarray[i] = Array(Float64, length(proposedtimes))
		for j = 1:length(bigoed.models)
			goodindices = (proposedmodelindices .== j)
			proposedobsarray[i][goodindices] = bigoed.models[j](vec(mcmcchain.samples[i, :]), bigoed.decisionparams[1], proposedlocations[goodindices], proposedtimes[goodindices])
		end
	end
	return proposedobsarray
end

function dobigoed(bigoed::BigOED, hakunamatata, numlikelihoods, numhorizons, numobsrealizations, acceptableprobabilityoffailure)
	bigdts = makebigdts(bigoed)
	maxfailureprobsarray = Array(Array{Float64, 1}, length(bigdts))
	horizonsarray = Array(Array{Float64, 1}, length(bigdts))
	for i = 1:length(bigdts)
		maxfailureprobsarray[i], horizonsarray[i], badlikelihoodparams = BIGUQ.getrobustnesscurve(bigdts[i], hakunamatata, numlikelihoods; numhorizons=numhorizons)
		println("Initial decision $i robustness curve:")
		printresults(maxfailureprobsarray[i], horizonsarray[i], badlikelihoodparams)
	end
	initialdecision = makedecision(maxfailureprobsarray, horizonsarray, acceptableprobabilityoffailure; robustnesspenalty=bigoed.robustnesspenalty)
	println("Initial decision: $initialdecision")
	decisionprobabilities = zeros(length(bigoed.proposedlocations), length(bigoed.decisionparams))
	iterationscomplete = 0
	for i = 1:length(bigoed.proposedlocations)#iterate through each possible data collection effort
		proposedobsarray = generateproposedobs(bigoed, i, numobsrealizations)
		for j = 1:numobsrealizations#iterate through each realization of the proposed observations
			bigdts = makebigdts(bigoed, i, proposedobsarray[j])
			for k = 1:length(bigdts)
				maxfailureprobsarray[k], horizonsarray[k], throwaway = getrobustnesscurve(bigdts[k], hakunamatata, numlikelihoods; numhorizons=numhorizons)
			end
			decision = makedecision(maxfailureprobsarray, horizonsarray, acceptableprobabilityoffailure; robustnesspenalty=bigoed.robustnesspenalty)
			decisionprobabilities[i, decision] += 1
			iterationscomplete += 1
		end
		f = open("progress.txt", "w")
		write(f, "$(iterationscomplete / length(bigoed.proposedlocations) / numobsrealizations)\n")
		close(f)
	end
	decisionprobabilities /= numobsrealizations
	println("Decision probabilities:\n$decisionprobabilities")
	return decisionprobabilities
end
