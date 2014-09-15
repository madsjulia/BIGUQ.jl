import MCMC
import Wells
import Optim
import ForwardDiff
import PyCall
@PyCall.pyimport pyDOE as doe

type Biguq
	model::Function
	makeloglikelihood::Function#we give it a set of likelihood parameters, and it gives us a conditional likelihood function 
	logprior::Function#the function encoding our prior beliefs
	nominalparams#nominal parameters for the model
	#now include functions that tell us about the infogap uncertainty model
	likelihoodparamsmin::Function#gives us the minimums of the likelihood params as a function of the horizon of uncertainty
	likelihoodparamsmax::Function#gives us the maximums of the likelihood params as a function of the horizon of uncertainty
	#now include a function that tells us whether the performance goal is satisfied -- this function includes information about the model uncertainty
	performancegoalsatisfied::Function#tells us whether the performance goal is satisfied as a function of the model output and the horizon of uncertainty
end

function getmcmcchain(biguq::Biguq, likelihoodparams; steps=int(1e5), burnin=int(1e4))
	loglikelihood = biguq.makeloglikelihood(likelihoodparams)
	mcmcmodel = MCMC.model(params -> biguq.logprior(params) + loglikelihood(params), init=biguq.nominalparams)
	rmw = MCMC.RWM(0.1)
	smc = MCMC.SerialMC(steps=steps, burnin=burnin)
	mcmcchain = MCMC.run(mcmcmodel, rmw, smc)
	return mcmcchain
end

function getfailureprobability(biguq::Biguq, horizon::Number, mcmcchain::MCMC.MCMCChain)
	failures = 0
	for sample in mcmcchain.samples
		if !biguq.performancegoalsatisfied(sample, horizon)
			failures += 1
		end
	end
	retval = failures / size(mcmcchain.samples)[1]
	#println("$horizon $retval")
	return retval
end

function getfailureprobabilities(biguq::Biguq, horizons::Vector, likelihoodparams)
	mcmcchain = getmcmcchain(biguq, likelihoodparams)
	results = similar(horizons)
	i = 1
	for horizon in horizons
		results[i] = getfailureprobability(biguq, horizon, mcmcchain)
		i += 1
	end
	return results
end

function inbox(x, mins, maxs)
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
	failureprobs = pmap(lparams -> getfailureprobabilities(biguq, horizons, lparams), likelihoodparams)
	maxfailureprobs = zeros(numhorizons)
	for i = 1:numlikelihoods
		for k = likelihoodhorizonindices[i]:numhorizons
			if failureprobs[i][k] > maxfailureprobs[k]
				maxfailureprobs[k] = failureprobs[i][k]
			end
		end
	end
	return maxfailureprobs
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

function test(biguq)
	return getrobustnesscurve(biguq, 10, 1000)
end

biguq1 = getbiguq1()
failureprobs = test(biguq1)
println("failureprobs: $failureprobs")

#times = linspace(1, 30, 30) * 24 * 3600
#deltaheads = zeros(30)
#Qw = .1 #m^3/sec
#K1 = 1e-3 #m/sec -- pervious
#K2 = 1e-5 #m/sec -- semi-pervious
#L1 = 100 #m
#L2 = 200 #m
#Sc1 = 7e-5 #m^-1 -- dense, sandy gravel
#Sc2 = 1e-5 #m^-1 -- fissured rock
#ra = .1 #m
#R = 100 #m
#omega = 1e3 #no resistance
#deltah = 0 #m
#r1 = 50 #m
#function loglikelihood(params)
#	Qw = params[1]
#	K1 = params[2]
#	K2 = params[3]
#	L1 = params[4]
#	L2 = params[5]
#	Sc1 = params[6]
#	Sc2 = params[7]
#	ra = params[8]
#	R = params[9]
#	omega = params[10]
#	deltah = params[11]
#	r1 = params[12]
#	#lp = logprior(K1, K2, L1, L2, Sci1, Sci2, ra, R, 
#	if K1 < 0 || K2 < 0 || L1 < 0 || L2 < 0 || Sc1 < 0 || Sc2 < 0 || ra < 0 || R < 0 || omega < 0 || r1 < 0
#		return -Inf
#	end
#	avcideltaheads = map(t -> Wells.avcideltahead(Qw, K1, K2, L1, L2, Sc1, Sc2, ra, R, omega, deltah, r1, t), times)
#	v = deltaheads - avcideltaheads
#	retval = -dot(v, v)
#	return retval
#end
#mymodel = MCMC.model(v -> -dot(v, v), grad=v -> -2 * v, init=ones(3))
#mychain = run(mymodel, MCMC.HMC(0.1), MCMC.SerialMC(steps=100000, burnin=10000))
#params0 = [Qw, K1, K2, L1, L2, Sc1, Sc2, ra, R, omega, deltah, r1]
##mymodel = MCMC.model(v -> -dot(v, v), init=params0)
#mymodel = MCMC.model(loglikelihood, init=params0)
#rmw = MCMC.RWM(0.1)
#smc = MCMC.SerialMC(steps=int(1e5), burnin=int(1e4))
#mychain = MCMC.run(mymodel, rmw, smc)
#MCMC.describe(mychain)
