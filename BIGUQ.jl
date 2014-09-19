import MCMC
import Wells
import Optim
import ForwardDiff
import PyCall
@PyCall.pyimport pyDOE as doe # called in getrobustnesscurve

type Biguq
	model::Function
        makeloglikelihood::Function # we give it a set of likelihood parameters, and it gives us a conditional likelihood function. That is, it gives us a function of the parameters that returns the likelihood of the data given the parameters
        logprior::Function # the function encoding our prior beliefs
        nominalparams # nominal parameters for the model
        # now include functions that tell us about the infogap uncertainty model
        likelihoodparamsmin::Function # gives us the minimums of the likelihood params as a function of the horizon of uncertainty
        likelihoodparamsmax::Function # gives us the maximums of the likelihood params as a function of the horizon of uncertainty
        # now include a function that tells us whether the performance goal is satisfied -- this function includes information about the model uncertainty
        performancegoalsatisfied::Function # tells us whether the performance goal is satisfied as a function of the model output and the horizon of uncertainty
end

function getmcmcchain(biguq::Biguq, likelihoodparams; steps=int(1e5), burnin=int(1e4))
	loglikelihood = biguq.makeloglikelihood(likelihoodparams)
	lhoodgrad = ForwardDiff.forwarddiff_gradient(params -> biguq.logprior(params) + loglikelihood(params), Float64, n=size(biguq.nominalparams, 1))
	println(lhoodgrad([4.]))
	mcmcmodel = MCMC.model(params -> biguq.logprior(params) + loglikelihood(params), grad=lhoodgrad, init=biguq.nominalparams)
	rmw = MCMC.RWM(0.1)
	#rmw = MCMC.HMC(3, 0.1)
	smc = MCMC.SerialMC(steps=steps, burnin=burnin)
	mcmcchain = MCMC.run(mcmcmodel, rmw, smc)
	MCMC.describe(mcmcchain)
	println(MCMC.acceptance(mcmcchain))
	return mcmcchain
end

function getfailureprobability(biguq::Biguq, horizon::Number, mcmcchain::MCMC.MCMCChain) # called in getfailureprobabilities
	failures = 0
	for i = 1:size(mcmcchain.samples)[1]
	#for sample in mcmcchain.samples
		sample = reshape(mcmcchain.samples[i, :], size(mcmcchain.samples)[2])
		if !biguq.performancegoalsatisfied(sample, horizon)
			failures += 1
		end
	end
	retval = failures / size(mcmcchain.samples)[1]
	return retval
end

function getfailureprobabilities(biguq::Biguq, horizons::Vector, likelihoodparams::Vector) # called in getrobustnesscurve
	mcmcchain = getmcmcchain(biguq, likelihoodparams)
	results = similar(horizons)
	i = 1
	for horizon in horizons
		results[i] = getfailureprobability(biguq, horizon, mcmcchain)
		i += 1
	end
	return results
end

function inbox(x, mins, maxs) # called in getrobustnesscurve
	return all(map(<=, x, maxs)) && all(map(>=, x, mins))
end

function getrobustnesscurve(biguq::Biguq, hakunamatata::Number, numlikelihoods::Int64; numhorizons::Int64=100)
	minlikelihoodparams = biguq.likelihoodparamsmin(hakunamatata)
	maxlikelihoodparams = biguq.likelihoodparamsmax(hakunamatata)
	lhs = doe.lhs(size(minlikelihoodparams)[1], samples=numlikelihoods)
	likelihoodparams = similar(lhs)
	likelihoodhorizonindices = Array(Int64, numlikelihoods)
	horizons = linspace(0, hakunamatata, numhorizons)
	for i = 1:numlikelihoods
		for j = 1:size(minlikelihoodparams)[1]
			likelihoodparams[i, j] = minlikelihoodparams[j] + lhs[i, j] * (maxlikelihoodparams[j] - minlikelihoodparams[j])
			k = 1
			likelihoodhorizonindices[i] = numhorizons
			while k <= numlikelihoods
				if inbox(likelihoodparams[i], biguq.likelihoodparamsmin(horizons[k]), biguq.likelihoodparamsmax(horizons[k]))
					likelihoodhorizonindices[i] = k
					k = numlikelihoods + 1
				end
				k += 1
			end
		end
	end
	failureprobs = pmap(i -> getfailureprobabilities(biguq, horizons, reshape(likelihoodparams[i, :], size(likelihoodparams)[2])), 1:size(likelihoodparams)[1])
	maxfailureprobs = zeros(numhorizons)
	badlikelihoodparams = Array(typeof(likelihoodparams[1]), numhorizons)
	for i = 1:numhorizons
		badlikelihoodparams[i, :] = biguq.likelihoodparamsmin(0.)
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

function getbiguq1()
	function model(params)
		k = params[1]
		return k * 2
	end
	function makeloglikelihood(likelihoodparams)
		N = likelihoodparams[1]
		#return params -> -(abs(params[1] * 1 - 1.5)) ^ N
		return params -> (params[1] <= N ? 0. : -Inf)
	end
	function logprior(params)
		k = params[1]
		if k > 0 && k < 10
			return 0.
		else
			return -Inf
		end
	end
	nominalparams = [2 / 3]
	function likelihoodparamsmin(horizon)
		[max(nominalparams[1], (1 - horizon) * 2.)]
	end
	function likelihoodparamsmax(horizon)
		[(1 + horizon) * 2.]
	end
	function performancegoalsatisfied(params, horizon)
		return (1 + 0.001 * horizon) * model(params) < 4.2
	end
	biguq = Biguq(model, makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
	return biguq
end

function getbiguq2()
	function model(params::Vector)
		return params[1]
	end
	const data = 1 + .1 * randn(5)
	function makeloglikelihood(likelihoodparams::Vector)
		logvar = likelihoodparams[1]
		var = exp(logvar)
		return params -> -.5 * sum((data - params[1]) .^ 2) / var - logvar
	end
	nominalparams = [.5]
	function logprior(params::Vector)
		return -.5 * (params[1] - nominalparams[1]) ^ 2 / .01
	end
	function likelihoodparamsmin(horizon::Number)
		return [4 - horizon]
	end
	function likelihoodparamsmax(horizon::Number)
		return [4 + horizon]
	end
	function performancegoalsatisfied(params::Vector, horizon::Number)
		return model(params) < .9
	end
	biguq = Biguq(model, makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
end

function test(biguq::Biguq)
	numhorizons = 10
	@time maxfailureprobs, horizons, badlikelihoodparams = getrobustnesscurve(biguq, 10, 10; numhorizons=numhorizons)
	for i = 1:numhorizons
		println(horizons[i], ": ", maxfailureprobs[i], " -- ", badlikelihoodparams[i])
	end
end

#biguq1 = getbiguq1()
#failureprobs = test(biguq1)
biguq2 = getbiguq2()
test(biguq2)
