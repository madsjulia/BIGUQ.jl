type BigOED
	#makemodel::Function#a function that takes a set of parameters as an argument and makes a model function that takes two arrays: one of spatial coordinates and another of times. It returns an array containing the results.
	model::Function#a function that takes a vector of parameters, and two arrays: one of spatial coordinates, and another of times. It returns an array containing the results.
	data::Vector
	datalocations::Vector
	datatimes::Vector
	proposedlocations::Vector
	proposedtimes::Vector
	makeresidualdistribution::Function#takes the params from the info-gap model as well as the datalocations, datatimes, proposedlocations[i], proposedtimes[i]
	residualdistributionparamsmin::Function
	residualdistributionparamsmax::Function
	nominalparams#nominal parameters for the model
	performancegoalsatisfied::Function
	logprior::Function
end

function makebigdt(bigoed::BigOED)
	function makeloglikelihood(likelihoodparams::Vector)
		const residualdistribution = bigoed.makeresidualdistribution(likelihoodparams, bigoed.datalocations, bigoed.datatimes, [], [])
		function loglikelihood(params::Vector)
			results = bigoed.model(params, bigoed.datalocations, bigoed.datatimes)
			residuals = bigoed.data - results
			retval = Distributions.logpdf(residualdistribution, residuals)
			return retval
		end
	end
	return BIGUQ.BigDT(makeloglikelihood, bigoed.logprior, bigoed.nominalparams, bigoed.residualdistributionparamsmin, bigoed.residualdistributionparamsmax, bigoed.performancegoalsatisfied)
end
