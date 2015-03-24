module BIGUQ
import Lora
import ForwardDiff
import BlackBoxOptim

type Biguq
	#model::Function
	makeloglikelihood::Function # we give it a set of likelihood parameters, and it gives us a conditional likelihood function. That is, it gives us a function of the parameters that returns the likelihood of the data given the parameters
	logprior::Function # the function encoding our prior beliefs
	nominalparams # nominal parameters for the model
	# now include functions that tell us about the infogap uncertainty model
	likelihoodparamsmin::Function # gives us the minimums of the likelihood params as a function of the horizon of uncertainty
	likelihoodparamsmax::Function # gives us the maximums of the likelihood params as a function of the horizon of uncertainty
	# now include a function that tells us whether the performance goal is satisfied -- this function includes information about the model uncertainty
	performancegoalsatisfied::Function # tells us whether the performance goal is satisfied as a function of the model output and the horizon of uncertainty
end

function getmcmcchain(biguq::Biguq, likelihoodparams; steps=int(1e4), burnin=int(1e3), usederivatives=false)
	conditionalloglikelihood = biguq.makeloglikelihood(likelihoodparams)
	function loglikelihood(params)
		l1 = biguq.logprior(params)
		if l1 == -Inf
			return -Inf
		else
			return l1 + conditionalloglikelihood(params)
		end
	end
	if usederivatives
		loglikelihoodgrad = ForwardDiff.forwarddiff_gradient(loglikelihood, Float64, fadtype=:dual)
		mcmcmodel = Lora.model(loglikelihood, grad=loglikelihoodgrad, init=biguq.nominalparams)
		rmw = Lora.HMC(3, 1e-2)
	else
		mcmcmodel = Lora.model(loglikelihood, init=biguq.nominalparams)
		#rmw = Lora.RWM(1e-2)
		rmw = Lora.RAM(1e-0, 0.3)
	end
	smc = Lora.SerialMC(nsteps=steps, burnin=burnin)
	mcmcchain = Lora.run(mcmcmodel, rmw, smc)
	ess = Lora.ess(mcmcchain)
	if minimum(ess) < 10
		warn(string("Low effective sample size, ", ess, ", with likelihood params ", likelihoodparams))
	end
	return mcmcchain
end

function get_min_index_of_horizon_with_failure(biguq::Biguq, sample::Vector, horizons::Vector) # called in getfailureprobabilities
	if !biguq.performancegoalsatisfied(sample, horizons[1])
		return 1
	elseif biguq.performancegoalsatisfied(sample, horizons[end])
		return length(horizons) + 1
	elseif biguq.performancegoalsatisfied(sample, horizons[int(.5 * length(horizons))])
		return int(.5 * length(horizons)) + get_min_index_of_horizon_with_failure(biguq, sample, horizons[int(.5 * length(horizons)) + 1:end])
	else
		return get_min_index_of_horizon_with_failure(biguq, sample, horizons[1:int(.5 * length(horizons)) - 1])
	end
end

function getfailureprobabilities(biguq::Biguq, horizons::Vector, likelihoodparams::Vector) # called in getrobustnesscurve
	mcmcchain = getmcmcchain(biguq, likelihoodparams)
	failures = zeros(Int64, length(horizons))
	for i = 1:size(mcmcchain.samples, 1)
		sample = reshape(mcmcchain.samples[i, :], size(mcmcchain.samples, 2))
		minindex = get_min_index_of_horizon_with_failure(biguq, sample, horizons)
		for j = minindex:length(failures)
			failures[j] += 1
		end
	end
	return failures / size(mcmcchain.samples, 1)
end


function inbox(x, mins, maxs) # called in getrobustnesscurve
	return all(map(<=, x, maxs)) && all(map(>=, x, mins))
end

function getrobustnesscurve(biguq::Biguq, hakunamatata::Number, numlikelihoods::Int64; numhorizons::Int64=100)
	minlikelihoodparams = biguq.likelihoodparamsmin(hakunamatata)
	maxlikelihoodparams = biguq.likelihoodparamsmax(hakunamatata)
	likelihoodparams = BlackBoxOptim.Utils.latin_hypercube_sampling(minlikelihoodparams, maxlikelihoodparams, numlikelihoods)
	likelihoodhorizonindices = Array(Int64, numlikelihoods)#This stores the index of the smallest horizon of uncertainty containing the likelihood parameters
	horizons = linspace(0, hakunamatata, numhorizons)
	for i = 1:numlikelihoods
		k = 1
		likelihoodhorizonindices[i] = numhorizons
		while k <= numhorizons
			if inbox(likelihoodparams[i], biguq.likelihoodparamsmin(horizons[k]), biguq.likelihoodparamsmax(horizons[k]))
				likelihoodhorizonindices[i] = k
				k = numhorizons + 1#do this to kill the loop
			end
			k += 1
		end
	end
	likelihoodparams = [biguq.likelihoodparamsmin(0); likelihoodparams]#make sure the nominal case is in there
	likelihoodhorizonindices = [1; likelihoodhorizonindices]
	numlikelihoods += 1
	reshapedparams = map(i -> reshape(likelihoodparams[i, :], size(likelihoodparams, 2)), 1:size(likelihoodparams, 1))
	failureprobs = pmap(p -> getfailureprobabilities(biguq, horizons, p), reshapedparams)
	maxfailureprobs = zeros(numhorizons)
	badlikelihoodparams = Array(Any, numhorizons)
	for i = 1:numhorizons
		badlikelihoodparams[i] = biguq.likelihoodparamsmin(0.)
	end
	for i = 1:numlikelihoods
		for k = likelihoodhorizonindices[i]:numhorizons
			if failureprobs[i][k] > maxfailureprobs[k]
				maxfailureprobs[k] = failureprobs[i][k]
				badlikelihoodparams[k] = likelihoodparams[i]
			end
		end
	end
	return maxfailureprobs, horizons, badlikelihoodparams
end

end
