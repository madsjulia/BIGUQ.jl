type BigOED
	models::Array{Function, 1}#an array of functions that takes a vector of uncertain parameters, a vector of (certain) decision parameters, and two arrays: one of spatial coordinates, and another of times. It returns an array containing the results.
	#the different models represent different things that could be measured at different times/locations
	obs::Vector
	obslocations::Vector
	obstimes::Vector
	obsmodelindices::Array{Int64, 1}#an index that indicates which model would be used to represent the measurement found in the corresponding location of the obs array
	proposedlocations::Vector
	proposedtimes::Vector
	proposedindices::Array{Int64, 1}#say which model the proposed data collection corresponds to
	makeresidualdistribution::Function#takes the params from the info-gap model as well as the obslocations, obstimes, proposedlocations[i], proposedtimes[i]
	residualdistributionparamsmin::Function
	residualdistributionparamsmax::Function
	nominalparams::Vector#nominal parameters for the model
	performancegoalsatisfied::Function#a function that take parameters, decision parameters, and a horizon of uncertainty
	logprior::Function
	decisionparams::Array{Array{Float64, 1}, 1}#an array of decision parameter arrays representing different possible decisions
end

function makebigdts(bigoed::BigOED, proposalindex::Int64)
	function makeloglikelihood(likelihoodparams::Vector, decisionindex::Int64)
		const constlikelihoodparams = copy(likelihoodparams)
		const proposedlocations = (proposalindex > 0 ? bigoed.proposedlocations : [])
		const proposedtimes = (proposalindex > 0 ? bigoed.proposedtimes : [])
		const proposedindices = (proposalindex > 0 ? bigoed.proposedindices : [])
		const residualdistribution = bigoed.makeresidualdistribution(constlikelihoodparams, bigoed.obslocations, bigoed.obstimes, proposedlocations, proposedtimes, proposedindices)
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
