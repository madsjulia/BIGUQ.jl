# @everywhere using ArrayViews

type BigDT
	makeloglikelihood::Function # we give it a set of likelihood parameters, and it gives us a conditional likelihood function. That is, it gives us a function of the parameters that returns the likelihood of the data given the parameters
	logprior::Function # the function encoding our prior beliefs
	nominalparams::Vector # nominal parameters for the model
	# now include functions that tell us about the infogap uncertainty model
	likelihoodparamsmin::Function # gives us the minimums of the likelihood params as a function of the horizon of uncertainty
	likelihoodparamsmax::Function # gives us the maximums of the likelihood params as a function of the horizon of uncertainty
	# now include a function that tells us whether the performance goal is satisfied -- this function includes information about the model uncertainty
	performancegoalsatisfied::Function # tells us whether the performance goal is satisfied as a function of the model parameters and the horizon of uncertainty
	gethorizonoffailure::Function
	useperformancegoalsatisfied::Bool
	function BigDT(makeloglikelihood::Function, logprior::Function, nominalparams::Vector, likelihoodparamsmin::Function, likelihoodparamsmax::Function, performancegoalsatisfied::Function)
		return new(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied, ()->1, true)
	end
	function BigDT(makeloglikelihood::Function, logprior::Function, nominalparams::Vector, likelihoodparamsmin::Function, likelihoodparamsmax::Function, performancegoalsatisfied::Function, gethorizonoffailure::Function)
		return new(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied, gethorizonoffailure, false)
	end
end

"Get MCMC chain"
#function getmcmcchain(bigdt::BigDT, likelihoodparams; steps=10 ^ 2, burnin=10, numwalkers=10 ^ 2)
function getmcmcchain(bigdt::BigDT, likelihoodparams; steps=3, burnin=2, numwalkers=8)
	conditionalloglikelihood = bigdt.makeloglikelihood(likelihoodparams)
	function loglikelihood(params)
		l1 = bigdt.logprior(params)
		if l1 == -Inf
			return -Inf
		else
			return l1 + conditionalloglikelihood(params)
		end
	end
	burninchain, burninllhoodvals = Mads.emcee(loglikelihood, numwalkers, broadcast(+, bigdt.nominalparams, 1e-6 * randn(length(bigdt.nominalparams), numwalkers)), burnin, 1)
	chain, llhoodvals = Mads.emcee(loglikelihood, numwalkers, broadcast(+, bigdt.nominalparams, 1e-6 * randn(length(bigdt.nominalparams), numwalkers)), steps, 1)
	return Mads.flattenmcmcarray(chain, llhoodvals)
end

function get_min_index_of_horizon_with_failure(bigdt::BigDT, sample::Vector, horizons::Vector) # called in getfailureprobabilities
	if bigdt.useperformancegoalsatisfied
		return get_min_index_of_horizon_with_failure(bigdt.performancegoalsatisfied, sample, horizons)
	else
		return get_min_index_of_horizon_with_failure(sample, horizons, bigdt.gethorizonoffailure)
	end
end

function get_min_index_of_horizon_with_failure(sample::Vector, horizons::Vector, gethorizonoffailure::Function)
	horizonoffailure = gethorizonoffailure(sample)
	maxindex = length(horizons)
	minindex = 1
	while maxindex > minindex + 1
		midindex = round(Int, .5 * (maxindex + minindex))
		if horizonoffailure == horizons[midindex]
			return midindex
		elseif horizonoffailure < horizons[midindex]
			maxindex = midindex
		else
			minindex = midindex
		end
	end
	if horizonoffailure > horizons[maxindex]
		return maxindex
	else
		return minindex
	end
end

function get_min_index_of_horizon_with_failure(performancegoalsatisfied::Function, sample::Vector, horizons::Vector)
	if !performancegoalsatisfied(sample, horizons[1])
		return 1
	elseif performancegoalsatisfied(sample, horizons[end])
		return length(horizons) + 1
	elseif performancegoalsatisfied(sample, horizons[round(Int, .5 * length(horizons))])
		return round(Int, .5 * length(horizons)) + get_min_index_of_horizon_with_failure(performancegoalsatisfied, sample, horizons[round(Int, .5 * length(horizons)) + 1:end])
	else
		return get_min_index_of_horizon_with_failure(performancegoalsatisfied, sample, horizons[1:round(Int, .5 * length(horizons)) - 1])
	end
end

#! Get failure probablities using Markov Chain Monte Carlo
function getfailureprobabilities(bigdt::BigDT, horizons::Vector, likelihoodparams::Vector) # called in getrobustnesscurve
	mcmcchain, _ = getmcmcchain(bigdt, likelihoodparams)
	failures = zeros(Int64, length(horizons))
	for i = 1:size(mcmcchain, 2)
		sample = mcmcchain[:, i]
		minindex = get_min_index_of_horizon_with_failure(bigdt, sample, horizons)
		for j = minindex:length(failures)
			failures[j] += 1
		end
	end
	return failures / size(mcmcchain, 2)
end

#! Make getfailureprobablities function using Latin Hypercube Sampling
function makegetfailureprobabilities_mc(modelparams::Matrix, origloglikelihoods=zeros(size(modelparams, 2)))
	const nummodelparams = size(modelparams, 2)

	return (bigdt::BigDT, horizons::Vector, likelihoodparams::Vector) -> begin
		const conditionalloglikelihood = bigdt.makeloglikelihood(likelihoodparams)
		function loglikelihood(params)
			l1 = bigdt.logprior(params)
			if l1 == -Inf
				return -Inf
			else
				return l1 + conditionalloglikelihood(params)
			end
		end

		sumweights = 0.
		failures = zeros(Float64, length(horizons))
		loglikelihoods = Array(Float64, nummodelparams)
		weights = Array(Float64, nummodelparams)
		for i = 1:nummodelparams
			params_i = modelparams[:,i]
			loglikelihoods[i] = loglikelihood(params_i) - origloglikelihoods[i]
		end
		maxloglikelihood = maximum(loglikelihoods)
		for i = 1:nummodelparams
			params_i = modelparams[:, i]
			weights[i] = exp(loglikelihoods[i] - maxloglikelihood)
			wij = exp(loglikelihoods[i] - maxloglikelihood)
			sumweights += wij
			minindex = get_min_index_of_horizon_with_failure(bigdt, params_i, horizons)
			for j = minindex:length(failures)
				failures[j] += wij
			end
		end
		if sumweights == 0.
			error("All samples have zero weight. likelihoodparams: $likelihoodparams")
		elseif sumweights == 1.
			warn("All or nearly all the weight was in one sample. likelihoodparams: $likelihoodparams")
		end
		return failures / sumweights
	end
end

function inbox(x, mins, maxs) # called in getrobustnesscurve
	return reduce(&, map(<=, x, maxs)) && reduce(&, map(>=, x, mins))
end

#! BigDT robustness curve
#!
#! \param bigdt BigDT object
#! \param hakunamatata Maximum horizon of uncertainity that is relevant
#! \param numlikelihoods Number of likelihood params to sample from the likelihood space
#! \param getfailureprobfnct Function for calculating failure probabilities
#! \param numhorizons Number of horizons of uncertainty
function getrobustnesscurve(bigdt::BigDT, hakunamatata::Number, numlikelihoods::Int64; getfailureprobfnct::Function=getfailureprobabilities, numhorizons::Int64=100, likelihoodparams::Matrix=zeros(0, 0))
	if length(likelihoodparams) == 0
		minlikelihoodparams = bigdt.likelihoodparamsmin(hakunamatata)
		maxlikelihoodparams = bigdt.likelihoodparamsmax(hakunamatata)
		likelihoodparams = BlackBoxOptim.Utils.latin_hypercube_sampling(map(Float64, minlikelihoodparams), map(Float64, maxlikelihoodparams), numlikelihoods)
	end
	horizons = collect(linspace(0, hakunamatata, numhorizons))

	# find `likelihoodhorizonindices`, or the index of the smallest horizon of uncertainty containing the parameters
	likelihoodhorizonindices = fill(numhorizons, numlikelihoods)
	for i = 1:numlikelihoods
		k = 1
		while k <= numhorizons
			if inbox(likelihoodparams[:, i], bigdt.likelihoodparamsmin(horizons[k]), bigdt.likelihoodparamsmax(horizons[k]))
				likelihoodhorizonindices[i] = k;
				break;
			end
			k += 1;
		end
	end

	temp = copy(bigdt.likelihoodparamsmin(0))
	likelihoodparams = [temp likelihoodparams] # make sure the nominal case is in there
	likelihoodhorizonindices = [1; likelihoodhorizonindices]
	numlikelihoods += 1
	likelihood_colvecs = [likelihoodparams[:,i] for i=1:size(likelihoodparams, 2)]
	#failureprobs = RobustPmap.rpmap(p->getfailureprobfnct(bigdt, horizons, p), likelihood_colvecs; t=Array{Float64, 1})
	failureprobs = map(p->getfailureprobfnct(bigdt, horizons, p), likelihood_colvecs)
	maxfailureprobs = zeros(numhorizons)

	badlikelihoodparams = Array(Array{Float64, 1}, numhorizons)
	for i = 1:numhorizons
		badlikelihoodparams[i] = bigdt.likelihoodparamsmin(0.)[1:end]
	end

	for i = 1:numlikelihoods
		for k = likelihoodhorizonindices[i]:numhorizons
			if failureprobs[i][k] > maxfailureprobs[k]
				maxfailureprobs[k] = failureprobs[i][k]
				badlikelihoodparams[k] = likelihoodparams[:, i]
			end
		end
	end
	return maxfailureprobs, horizons, badlikelihoodparams
end

function printresults(maxfailureprobs, horizons, badlikelihoodparams)
	for i = 1:length(horizons)
		println(horizons[i], ",", maxfailureprobs[i], ",", badlikelihoodparams[i])
	end
end

function dataframeresults(maxfailureprobs, horizons, badlikelihoodparams)
	return DataFrames.DataFrame(horizon=horizons, maxfailureprob=maxfailureprobs, badlikelihoodparams=badlikelihoodparams)
end

function getrobustness(maxfailureprobs, horizons, acceptableprobabilityoffailure)
	if acceptableprobabilityoffailure <= maxfailureprobs[1]
		return horizons[1]
	elseif acceptableprobabilityoffailure >= maxfailureprobs[end]
		return horizons[end]
	end
	i = 2
	while maxfailureprobs[i] < acceptableprobabilityoffailure
		i += 1
	end
	#do linear interpolation and find where the line hits acceptableprobabilityoffailure
	return horizons[i - 1] + (acceptableprobabilityoffailure - maxfailureprobs[i - 1]) * (horizons[i] - horizons[i - 1]) / (maxfailureprobs[i] - maxfailureprobs[i - 1])
end

function makedecision(bigdts::Array{BigDT, 1}, acceptableprobabilityoffailure, hakunamatata, numlikelihoods, numhorizons; robustnesspenalty=zeros(length(bigdts)))
	maxfailureprobsarray = Array(Array{Float64, 1}, length(bigdts))
	horizonsarray = Array(Array{Float64, 1}, length(bigdts))
	for i = 1:length(bigdts)
		maxfailureprobsarray[i], horizonsarray[i], throwaway = getrobustnesscurve(bigdts[i], hakunamatata, numlikelihoods; numhorizons=numhorizons)
	end
	return makedecision(maxfailureprobsarray, horizonsarray, acceptableprobabilityoffailure; robustnesspenalty=robustnesspenalty)
end

function makedecision(maxfailureprobsarray::Array{Array{Float64, 1}}, horizonsarray::Array{Array{Float64, 1}}, acceptableprobabilityoffailure; robustnesspenalty=zeros(length(maxfailureprobsarray)))
	robustnesses = map(i->getrobustness(maxfailureprobsarray[i], horizonsarray[i], acceptableprobabilityoffailure) - robustnesspenalty[i], 1:length(maxfailureprobsarray))
	decision = indmax(robustnesses)
	return decision
end
