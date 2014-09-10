import MCMC
import Wells
import Optim
import ForwardDiff

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

function getfailureprobability(biguq::Biguq, horizon::Number, likelihoodparams)
	loglikelihood = biguq.makeloglikelihood(likelihoodparams)
	mcmcmodel = MCMC.model(params -> biguq.logprior(params) + loglikelihood(params), init=biguq.nominalparams)
	rmw = MCMC.RWM(0.1)
	smc = MCMC.SerialMC(steps=int(1e5), burnin=int(1e4))
	#println("likeparams: $likelihoodparams")
	mcmcchain = MCMC.run(mcmcmodel, rmw, smc)
	#MCMC.describe(mcmcchain)
	failures = 0
	for sample in mcmcchain.samples
		if !biguq.performancegoalsatisfied(sample, horizon)
			failures += 1
		end
	end
	retval = failures / size(mcmcchain.samples)[1]
	println("$horizon $retval")
	return retval
end

function getmaxfailureprobabilities(biguq::Biguq, horizons::Array{Float64, 1})
	results = Array(Float64, size(horizons)[1])
	i = 1
	for horizon in horizons
		l = biguq.likelihoodparamsmin(horizon)
		u = biguq.likelihoodparamsmax(horizon)
		x0 = biguq.likelihoodparamsmin(0)
		#println("x0: $x0")
		#println("l: $l")
		#println("u: $u")
		#df = DifferentiableFunction(x -> -getfailureprobability(biguq, horizon, x))
		#results[i] = fminbox(df, x0, l, u)
		#g! = ForwardDiff.forwarddiff_gradient!(x -> getfailureprobability(biguq, horizon, x), Float64, n=size(biguq.nominalparams)[1])
		h = 0.01
#		function getfailureprobability(a, b, x)
#			return x[1]
#		end
		function myf(storage::Vector, x)
			#println("x: $x")
			#g!(x, storage)
			retval = -getfailureprobability(biguq, horizon, x)
			for j = 1:size(storage)[1]
				xpdx = copy(x)
				xpdx[j] += h
				fval = -getfailureprobability(biguq, horizon, xpdx)
				storage[j] = (fval - retval) / h
			end
			#println("retval: $retval")
			#println("gradient: $storage")
			return retval
		end
		x, fval, fcount, converged = Optim.fminbox(myf, x0, l, u)
		println("worstx: $x")
		results[i] = getfailureprobability(biguq, horizon, x)
		i += 1
	end
	println(results)
	return results
end

function getbiguq1()
	function model(params)
		k = params[1]
		return k * 2
	end
	function makeloglikelihood(likelihoodparams)
		N = likelihoodparams[1]
		#return params -> -(abs(params[1] * 1 - 1.5)) ^ N
		return params -> (params[1] < N ? 0. : -Inf)
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
		[max(.5, (1 - horizon) * 2.)]
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
	return getmaxfailureprobabilities(biguq, [.1, 1., 10.])
end

biguq1 = getbiguq1()
test(biguq1)

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
