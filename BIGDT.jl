type BigDT
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

function getmcmcchain(bigdt::BigDT, likelihoodparams; steps=int(1e5), burnin=int(1e4), usederivatives=false)
	conditionalloglikelihood = bigdt.makeloglikelihood(likelihoodparams)
	function loglikelihood(params)
		l1 = bigdt.logprior(params)
		if l1 == -Inf
			return -Inf
		else
			return l1 + conditionalloglikelihood(params)
		end
	end
	if usederivatives
		loglikelihoodgrad = ForwardDiff.forwarddiff_gradient(loglikelihood, Float64, fadtype=:dual)
		mcmcmodel = Lora.model(loglikelihood, grad=loglikelihoodgrad, init=bigdt.nominalparams)
		rmw = Lora.HMC(3, 1e-2)
	else
		mcmcmodel = Lora.model(loglikelihood, init=bigdt.nominalparams)
		#rmw = Lora.RWM(1e-2)
		rmw = Lora.RAM(1e-1, 0.3)
	end
	smc = Lora.SerialMC(nsteps=steps, burnin=burnin, thinning=10)
	mcmcchain = Lora.run(mcmcmodel, rmw, smc)
	#=
	Lora.describe(mcmcchain)
	println(mcmcchain)
	ess = Lora.ess(mcmcchain)
	if minimum(ess) < 10
		warn(string("Low effective sample size, ", ess, ", with likelihood params ", likelihoodparams))
	end
	=#
	return mcmcchain
end

function get_min_index_of_horizon_with_failure(bigdt::BigDT, sample::Vector, horizons::Vector) # called in getfailureprobabilities
	if !bigdt.performancegoalsatisfied(sample, horizons[1])
		return 1
	elseif bigdt.performancegoalsatisfied(sample, horizons[end])
		return length(horizons) + 1
	elseif bigdt.performancegoalsatisfied(sample, horizons[int(.5 * length(horizons))])
		return int(.5 * length(horizons)) + get_min_index_of_horizon_with_failure(bigdt, sample, horizons[int(.5 * length(horizons)) + 1:end])
	else
		return get_min_index_of_horizon_with_failure(bigdt, sample, horizons[1:int(.5 * length(horizons)) - 1])
	end
end

function getfailureprobabilities(bigdt::BigDT, horizons::Vector, likelihoodparams::Vector) # called in getrobustnesscurve
	mcmcchain = getmcmcchain(bigdt, likelihoodparams)
	failures = zeros(Int64, length(horizons))
	for i = 1:size(mcmcchain.samples, 1)
		sample = reshape(mcmcchain.samples[i, :], size(mcmcchain.samples, 2))
		minindex = get_min_index_of_horizon_with_failure(bigdt, sample, horizons)
		for j = minindex:length(failures)
			failures[j] += 1
		end
	end
	return failures / size(mcmcchain.samples, 1)
end


function inbox(x, mins, maxs) # called in getrobustnesscurve
	return all(map(<=, x, maxs)) && all(map(>=, x, mins))
end

function getrobustnesscurve(bigdt::BigDT, hakunamatata::Number, numlikelihoods::Int64; numhorizons::Int64=100)
	minlikelihoodparams = bigdt.likelihoodparamsmin(hakunamatata)
	maxlikelihoodparams = bigdt.likelihoodparamsmax(hakunamatata)
	likelihoodparams = BlackBoxOptim.Utils.latin_hypercube_sampling(minlikelihoodparams, maxlikelihoodparams, numlikelihoods)
	likelihoodhorizonindices = Array(Int64, numlikelihoods)#This stores the index of the smallest horizon of uncertainty containing the likelihood parameters
	horizons = linspace(0, hakunamatata, numhorizons)
	for i = 1:numlikelihoods
		k = 1
		likelihoodhorizonindices[i] = numhorizons
		while k <= numhorizons
			if inbox(likelihoodparams[i], bigdt.likelihoodparamsmin(horizons[k]), bigdt.likelihoodparamsmax(horizons[k]))
				likelihoodhorizonindices[i] = k
				k = numhorizons + 1#do this to kill the loop
			end
			k += 1
		end
	end
	temp = copy(bigdt.likelihoodparamsmin(0))
	likelihoodparams = [temp likelihoodparams]#make sure the nominal case is in there
	likelihoodhorizonindices = [1; likelihoodhorizonindices]
	numlikelihoods += 1
	reshapedparams = map(i -> reshape(likelihoodparams[i, :], size(likelihoodparams, 2)), 1:size(likelihoodparams, 1))
	failureprobs = pmap(p -> getfailureprobabilities(bigdt, horizons, p), reshapedparams)
	maxfailureprobs = zeros(numhorizons)
	badlikelihoodparams = Array(Array{Float64, 1}, numhorizons)
	for i = 1:numhorizons
		badlikelihoodparams[i] = bigdt.likelihoodparamsmin(0.)[1:end]
	end
	for i = 1:numlikelihoods
		for k = likelihoodhorizonindices[i]:numhorizons
			if failureprobs[i][k] > maxfailureprobs[k]
				maxfailureprobs[k] = failureprobs[i][k]
				if size(likelihoodparams, 2) == 1
					badlikelihoodparams[k] = [likelihoodparams[i]]
				else
					badlikelihoodparams[k] = vec(likelihoodparams[i, :])
				end
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
