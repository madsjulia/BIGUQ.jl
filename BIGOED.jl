type BigOED
	models::Array{Function, 1}#an array of functions that takes a vector of uncertain parameters, a vector of (certain) decision parameters, and two arrays: one of spatial coordinates, and another of times. It returns an array containing the results.
	#the different models represent different things that could be measured at different times/locations
	obs::Vector
	obslocations::Vector
	obstimes::Vector
	obsmodelindices::Array{Int64, 1}#an index that indicates which model would be used to represent the measurement found in the corresponding location of the obs array
	proposedlocations::Vector
	proposedtimes::Vector
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
		const residualdistribution = bigoed.makeresidualdistribution(constlikelihoodparams, bigoed.obslocations, bigoed.obstimes, proposedlocations, proposedtimes, proposedmodelindices)
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

macro dpt(ex)
	s = string(ex)
	return :( println($s, ": ", $ex, " (", string(typeof($ex)), ")") )
end

function makebigdts(bigoed::BigOED, proposedindex, proposedobs)
	function makeloglikelihood(likelihoodparams::Vector, decisionindex::Int64)
		const constlikelihoodparams = copy(likelihoodparams)
		const proposedlocations = bigoed.proposedlocations[proposedindex]
		const proposedtimes = bigoed.proposedtimes[proposedindex]
		const proposedmodelindices = bigoed.proposedmodelindices[proposedindex]
		const residualdistribution = bigoed.makeresidualdistribution(constlikelihoodparams, bigoed.obslocations, bigoed.obstimes, proposedlocations, proposedtimes, proposedmodelindices)
		function loglikelihood(params::Vector)
			results = Array(Float64, length(bigoed.obs) + length(proposedtimes))
			for i = 1:length(bigoed.models)
				goodindicesshort = (bigoed.obsmodelindices .== i)
				goodindiceslong = [goodindicesshort; fill(false, length(proposedmodelindices))]
				#=
				@dpt i
				@dpt proposedmodelindices
				@dpt length(results)
				@dpt length(goodindiceslong)
				@dpt goodindiceslong
				@dpt length(goodindicesshort)
				@dpt goodindicesshort
				@dpt results[goodindiceslong]
				@dpt bigoed.models[i](params, bigoed.decisionparams[decisionindex], bigoed.obslocations[goodindicesshort], bigoed.obstimes[goodindicesshort])
				=#
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
