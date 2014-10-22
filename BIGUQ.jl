module BIGUQ
import MCMC
import ForwardDiff
import BlackBoxOptim
using Gadfly

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

function getmcmcchain(biguq::Biguq, likelihoodparams; steps=int(1e5), burnin=int(1e4), usederivatives=false)
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
		#loglikelihoodgrad = ForwardDiff.forwarddiff_gradient(loglikelihood, Float64, n=size(biguq.nominalparams, 1))
		loglikelihoodgrad = ForwardDiff.forwarddiff_gradient(loglikelihood, Float64, fadtype=:dual)
		mcmcmodel = MCMC.model(loglikelihood, grad=loglikelihoodgrad, init=biguq.nominalparams)
		rmw = MCMC.HMC(3, 1e-2)
	else
		mcmcmodel = MCMC.model(loglikelihood, init=biguq.nominalparams)
		#rmw = MCMC.RWM(1e-2)
		rmw = MCMC.RAM(1e-0, 0.3)
	end
	smc = MCMC.SerialMC(steps=steps, burnin=burnin)
	mcmcchain = MCMC.run(mcmcmodel, rmw, smc)
	ess = MCMC.ess(mcmcchain)
	if min(ess...) < 10
		warn(string("Low effective sample size, ", ess, ", with likelihood params ", likelihoodparams))
	end
	#=
	MCMC.describe(mcmcchain)
	dh1 = Array(Float64, size(mcmcchain.samples, 1))
	dh2 = Array(Float64, size(mcmcchain.samples, 1))
	for i = 1:size(mcmcchain.samples, 1)
		sample = reshape(mcmcchain.samples[i, :], size(mcmcchain.samples, 2))
		dh1[i], dh2[i] = biguq.model(sample)
	end
	println("Upper:")
	println("\tmin:  ", min(dh1...))
	println("\tmean: ", mean(dh1))
	println("\tmax:  ", max(dh1...))
	println("Lower:")
	println("\tmin:  ", min(dh2...))
	println("\tmean: ", mean(dh2))
	println("\tmax:  ", max(dh2...))
	println("acceptance: ", MCMC.acceptance(mcmcchain))
	=#
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
	@time mcmcchain = getmcmcchain(biguq, likelihoodparams)
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
	#failureprobs = pmap(i -> getfailureprobabilities(biguq, horizons, reshape(likelihoodparams[i, :], size(likelihoodparams, 2))), 1:size(likelihoodparams, 1))
	#failureprobs = map(i -> getfailureprobabilities(biguq, horizons, reshape(likelihoodparams[i, :], size(likelihoodparams, 2))), 1:size(likelihoodparams, 1))
	#=
	layers = Array(Any, length(failureprobs) + 1)
	for i in 1:length(failureprobs)
		fps = copy(failureprobs[i])
		for j = 1:likelihoodhorizonindices[i] - 1
			fps[j] = 0
		end
		#println(likelihoodparams[i], " ", fps)
		layers[i] = layer(x=horizons, y=fps, Geom.line)
	end
	=#
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
	#=
	layers[end] = layer(x=horizons, y=maxfailureprobs, Geom.point)
	p = plot(layers...)
	draw(PNG("all-$(length(horizons)).png", 800px, 600px), p)
	run(`open all-$(length(horizons)).png`)
	=#
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
	biguq = Biguq(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
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
	biguq = Biguq(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
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
#biguq2 = getbiguq2()
#test(biguq2)
end
