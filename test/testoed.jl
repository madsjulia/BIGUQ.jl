module testoed
import BIGUQ
import Anasol
import Distributions
import ReusableFunctions

#@everywhere begin
#function makebigoed1()
	#srand(0)
	function innermodel(p::Vector)
		local params = p[1:9]
		local decisionparams = p[10:10]
		local numxs = round(Int, (length(p) - 10) / 2)
		local xs = p[11:10 + numxs]
		local ts = p[11 + numxs:10 + 2 * numxs]
		#x::Vector,tau,x01,sigma01,v1,sigma1,x02,sigma02,v2,sigma2
		local x01 = params[1]
		local sigma01 = params[2]
		local v1 = params[3]
		local sigma1 = params[4]
		local x02 = params[5]
		local sigma02 = params[6]
		local v2 = params[7]
		local sigma2 = params[8]
		local mass = params[9]
		local lambda = decisionparams[1]
		local result = Array{Float64}(length(xs))
		for i = 1:length(xs)
			#the background concentration is 5
			result[i] = 5. + mass * exp(-lambda * max(0., ts[i] - 4.5)) * Anasol.long_bb_dd_ii(xs[i], ts[i],
				x01, sigma01, v1, sigma1, 0.5, 1,
				x02, sigma02, v2, sigma2, 0.5, 1)
		end
		return result
	end
	#r3innermodel = ReusableFunctions.maker3function(innermodel)
	function model(params::Vector, decisionparams::Vector, xs::Vector, ts::Vector)
		#return r3innermodel([params[1:end]; decisionparams[1:end]; xs[1:end]; ts[1:end]])
		return innermodel([params[1:end]; decisionparams[1:end]; xs[1:end]; ts[1:end]])
	end
	#set up the "truth"
	x01 = 0.
	sigma01 = 1e-3
	v1 = 1e-1
	sigma1 = sqrt(1e-1)
	x02 = 0.
	sigma02 = 1e-3
	v2 = 1e-2
	sigma2 = sqrt(1e-2)
	mass = 1e2
	params = [x01, sigma01, v1, sigma1, x02, sigma02, v2, sigma2, mass]
	#f = makemodel(params)
	#these are the times and places where we have already collected data
	ts = [0.,1.,2.,0.,1.,2.,0.,1.,2.]
	xs = Array{Array{Float64, 1}}(9)
	xs[1] = [0., -.1]
	xs[2] = [0., -.1]
	xs[3] = [0., -.1]
	xs[4] = [0., .1]
	xs[5] = [0., .1]
	xs[6] = [0., .1]
	xs[7] = [-.1, 0.]
	xs[8] = [-.1, 0.]
	xs[9] = [-.1, 0.]
	noiselevel = 25.
	data = model(params, [0.], xs, ts)
	data += noiselevel * randn(length(data))
	proposedlocations = Array{Array{Array{Float64, 1}, 1}}(5)
	proposedlocations[1] = Array{Array{Float64, 1}}(4)
	proposedlocations[1][1:3] = xs[1:3:7]
	proposedlocations[1][4] = [.25, 0.]
	proposedlocations[2] = Array{Array{Float64, 1}}(4)
	proposedlocations[2][1:3] = xs[1:3:7]
	proposedlocations[2][4] = [.25, -.125]
	proposedlocations[3] = Array{Array{Float64, 1}}(4)
	proposedlocations[3][1:3] = xs[1:3:7]
	proposedlocations[3][4] = [.25, .125]
	proposedlocations[4] = Array{Array{Float64, 1}}(4)
	proposedlocations[4][1:3] = xs[1:3:7]
	proposedlocations[4][4] = [.125, 0.]
	proposedlocations[5] = Array{Array{Float64, 1}}(4)
	proposedlocations[5][1:3] = xs[1:3:7]
	proposedlocations[5][4] = [.375, 0.]
	proposedtimes = Array{Array{Float64, 1}}(5)
	proposedtimes[1] = [3., 3., 3., 3.]
	proposedtimes[2] = [3., 3., 3., 3.]
	proposedtimes[3] = [3., 3., 3., 3.]
	proposedtimes[4] = [3., 3., 3., 3.]
	proposedtimes[5] = [3., 3., 3., 3.]
	proposedmodelindices = Array{Array{Int64, 1}}(5)
	proposedmodelindices[1] = ones(Int, length(proposedlocations[1]))
	proposedmodelindices[2] = ones(Int, length(proposedlocations[2]))
	proposedmodelindices[3] = ones(Int, length(proposedlocations[3]))
	proposedmodelindices[4] = ones(Int, length(proposedlocations[4]))
	proposedmodelindices[5] = ones(Int, length(proposedlocations[5]))
	function rationalquadraticcovariance(d, sigma, alpha, k)
		nuggetvariance = 30.
		if d == 0.
			return nuggetvariance + sigma * (1. + (d * d) / (2 * alpha * k * k)) ^ (-alpha)
		else
			return sigma * (1. + (d * d) / (2 * alpha * k * k)) ^ (-alpha)
		end
	end
	function makeresidualdistribution(geostatparams, datalocations, datatimes, datamodelindices, proposedlocations, proposedtimes, proposedmodelindices)
		local sigma = geostatparams[1]
		local alpha = geostatparams[2]
		local k = geostatparams[3]
		alllocations = [datalocations; proposedlocations]
		alltimes = [datatimes; proposedtimes]
		local covmat = Array{Float64}((length(alllocations), length(alllocations)))
		for i = 1:length(alllocations)
			covmat[i, i] = rationalquadraticcovariance(0., sigma, alpha, k)
			for j = i+1:length(alllocations)
				h = sqrt(norm(alllocations[i] - alllocations[j]) ^ 2 + (alltimes[i] - alltimes[j]) ^ 2)
				covmat[i, j] = rationalquadraticcovariance(h, sigma, alpha, k)
				covmat[j, i] = covmat[i, j]
			end
		end
		return Distributions.MvNormal(covmat)
	end
	function residualdistributionparamsmin(horizonofuncertainty)
		local sigma = max(1., 100. * (1 - horizonofuncertainty))
		local alpha = max(.25, 2. - horizonofuncertainty)
		local k = max(.001, .1 * (1 - horizonofuncertainty))#k is like a length scale
		return [sigma, alpha, k]
	end
	function residualdistributionparamsmax(horizonofuncertainty)
		local sigma = 100. * (1 + horizonofuncertainty)
		local alpha = 2. + horizonofuncertainty
		local k = .1 * (1 + horizonofuncertainty)#k is like a length scale
		return [sigma, alpha, k]
	end
	nominalparams = params + sqrt(params) .* randn(length(params)) / 100
	ncompliancepoints = 10
	ncompliancetimes = 10
	compliancepoints = Array{Array{Float64, 1}}(ncompliancepoints * ncompliancetimes)
	compliancetimes = Array{Float64}(ncompliancepoints * ncompliancetimes)
	for i = 1:ncompliancetimes
		for j = 1:ncompliancepoints
			compliancepoints[(i - 1) * ncompliancepoints + j] = [.5, (j - .5 * ncompliancepoints) / ncompliancepoints]
			compliancetimes[(i - 1) * ncompliancepoints + j] = 4. + i
		end
	end
	compliancethreshold = 155.
	function performancegoalsatisfied(params::Vector, decisionparams::Vector, horizon::Number)
		results = model(params, decisionparams, compliancepoints, compliancetimes)
		results *= (1 + horizon)
		return !any(results .> compliancethreshold)
	end
	function gethorizonoffailure(params::Vector, decisionparams::Vector)
		results = model(params, decisionparams, compliancepoints, compliancetimes)
		minhorizonoffailure::Float64 = Inf
		horizonsoffailure = max(0., compliancethreshold ./ results - 1)
		return minimum(horizonsoffailure)
	end
	#=
	x01bounds = [-1., 1.]
	sigma01bounds = [1e-6, 1e-1]
	v1bounds = [1e-2, 5e-1]
	sigma1bounds = [sqrt(1e-2), sqrt(1e0)]
	x02bounds = [-1., 1.]
	sigma02bounds = [1e-6, 1e-1]
	v2bounds = [-5e-1, 5e-1]
	sigma2bounds = [sqrt(1e-4), sqrt(1e0)]
	massbounds = [1e1, 3e2]
	=#
	x01bounds = [-.05, .05]
	sigma01bounds = [7.5e-4, 1.5e-3]
	v1bounds = [7.5e-2, 1.5e-1]
	sigma1bounds = [sqrt(7.5e-2), sqrt(1.5e-1)]
	x02bounds = [-.05, .05]
	sigma02bounds = [7.5e-4, 1.5e-3]
	v2bounds = [-2e-2, 2e-2]
	sigma2bounds = [sqrt(7.5e-3), sqrt(1.5e-2)]
	massbounds = [7.5e1, 1.5e2]
	paramsmin = [x01bounds[1], sigma01bounds[1], v1bounds[1], sigma1bounds[1], x02bounds[1], sigma02bounds[1], v2bounds[1], sigma2bounds[1], massbounds[1]]
	paramsmax = [x01bounds[2], sigma01bounds[2], v1bounds[2], sigma1bounds[2], x02bounds[2], sigma02bounds[2], v2bounds[2], sigma2bounds[2], massbounds[2]]
	function logprior(params::Vector)
		if any(params .< paramsmin) || any(params .> paramsmax)
			return -Inf
		else
			return 1.
		end
	end
	decisionparams = Array{Array{Float64, 1}}(2)
	decisionparams[1] = zeros(1)
	decisionparams[2] = 0.2 * ones(1)
	robustnesspenalty = [0., .15]
	#=
	println(model(params, decisionparams[1], compliancepoints, compliancetimes))
	println(maximum(model(params, decisionparams[1], compliancepoints, compliancetimes)))
	println(model(params, decisionparams[2], compliancepoints, compliancetimes))
	println(maximum(model(params, decisionparams[2], compliancepoints, compliancetimes)))
	=#
	#return paramsmin, paramsmax, BIGUQ.BigOED([model], data, xs, ts, map(int, ones(length(data))), proposedlocations, proposedtimes, proposedmodelindices, makeresidualdistribution, residualdistributionparamsmin, residualdistributionparamsmax, nominalparams, performancegoalsatisfied, logprior, decisionparams, robustnesspenalty)
	#return paramsmin, paramsmax, BIGUQ.BigOED([model], data, xs, ts, ones(Int, length(data)), proposedlocations, proposedtimes, proposedmodelindices, makeresidualdistribution, residualdistributionparamsmin, residualdistributionparamsmax, nominalparams, performancegoalsatisfied, logprior, decisionparams, robustnesspenalty, gethorizonoffailure)
bigoed1 = BIGUQ.BigOED([model], data, xs, ts, ones(Int, length(data)), proposedlocations, proposedtimes, proposedmodelindices, makeresidualdistribution, residualdistributionparamsmin, residualdistributionparamsmax, nominalparams, performancegoalsatisfied, logprior, decisionparams, robustnesspenalty, gethorizonoffailure)
#end

end
