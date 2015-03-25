@everywhere import BIGUQ
@everywhere import Anasol

function makebigoed1()
	srand(0)
	#=
	function makemodel(params::Vector)
		#x::Vector,tau,x01,sigma01,v1,sigma1,x02,sigma02,v2,sigma2
		const x01 = params[1]
		const sigma01 = params[2]
		const v1 = params[3]
		const sigma1 = params[4]
		const x02 = params[5]
		const sigma02 = params[6]
		const v2 = params[7]
		const sigma2 = params[8]
		const mass = params[9]
		function f(xs::Vector, ts::Vector)
			const result = Array(Float64, length(xs))
			for i = 1:length(xs)
				result[i] = mass * Anasol.bb_dd_ii(xs[i], ts[i], x01, sigma01, v1, sigma1, x02, sigma02, v2, sigma2)
			end
			return result
		end
	end
	=#
	function model(params::Vector, xs::Vector, ts::Vector)
		#x::Vector,tau,x01,sigma01,v1,sigma1,x02,sigma02,v2,sigma2
		const x01 = params[1]
		const sigma01 = params[2]
		const v1 = params[3]
		const sigma1 = params[4]
		const x02 = params[5]
		const sigma02 = params[6]
		const v2 = params[7]
		const sigma2 = params[8]
		const mass = params[9]
		const result = Array(Float64, length(xs))
		for i = 1:length(xs)
			result[i] = mass * Anasol.bb_dd_ii(xs[i], ts[i], x01, sigma01, v1, sigma1, x02, sigma02, v2, sigma2)
		end
		return result
	end
	#set up the "truth"
	const x01 = 0.
	const sigma01 = 1e-3
	const v1 = 1e-1
	const sigma1 = sqrt(1e-1)
	const x02 = 0.
	const sigma02 = 1e-3
	const v2 = 1e-2
	const sigma2 = sqrt(1e-2)
	const mass = 1e2
	const params = [x01, sigma01, v1, sigma1, x02, sigma02, v2, sigma2, mass]
	#const f = makemodel(params)
	#these are the times and places where we have already collected data
	const ts = [0.,1.,2.,0.,1.,2.,0.,1.,2.]
	const xs = Array(Array{Float64, 1}, 9)
	xs[1] = [0., -.1]
	xs[2] = [0., -.1]
	xs[3] = [0., -.1]
	xs[4] = [0., .1]
	xs[5] = [0., .1]
	xs[6] = [0., .1]
	xs[7] = [-.1, 0.]
	xs[8] = [-.1, 0.]
	xs[9] = [-.1, 0.]
	const noiselevel = 25.
	data = model(params, xs, ts)
	data += noiselevel * randn(length(data))
	const proposedlocations = Array(Array{Array{Float64, 1}, 1}, 2)
	proposedlocations[1] = Array(Array{Float64, 1}, 4)
	proposedlocations[1][1:3] = xs[1:3:7]
	proposedlocations[1][4] = [-2., 1.]
	proposedlocations[2] = Array(Array{Float64, 1}, 4)
	proposedlocations[2][1:3] = xs[1:3:7]
	proposedlocations[2][4] = [-2.1, 1.1]
	const proposedtimes = Array(Array{Float64, 1}, 2)
	proposedtimes[1] = [3., 3., 3., 3.]
	proposedtimes[2] = [3., 3., 3., 3.]
	function rationalquadraticcovariance(d, sigma, alpha, k)
		return sigma * (1. + (d * d) / (2 * alpha * k * k)) ^ (-alpha)
	end
	function makeresidualdistribution(geostatparams, datalocations, datatimes, proposedlocations, proposedtimes)
		sigma = geostatparams[1]
		alpha = geostatparams[2]
		k = geostatparams[3]
		const alllocations = [datalocations; proposedlocations]
		const alltimes = [datatimes; proposedtimes]
		covmat = Array(Float64, (length(alllocations), length(alllocations)))
		for i = 1:length(alllocations)
			covmat[i, i] = rationalquadraticcovariance(0., sigma, alpha, k)
			for j = i+1:length(alllocations)
				h = sqrt(norm(alllocations[i] - alllocations[j]) ^ 2 + (alltimes[i] - alltimes[j]) ^ 2)
				covmat[i, j] = rationalquadraticcovariance(h, sigma, alpha, k)
			end
		end
		return Distributions.MvNormal(covmat)
	end
	function residualdistributionparamsmin(horizonofuncertainty)
		sigma = max(10., 100 - 100 * horizonofuncertainty)
		alpha = max(.25, 2. - horizonofuncertainty)
		k = max(.001, .1 - .1 * horizonofuncertainty)#k is like a length scale
		return [sigma, alpha, k]
	end
	function residualdistributionparamsmax(horizonofuncertainty)
		sigma = 100 + 100 * horizonofuncertainty
		alpha = 2. + horizonofuncertainty
		k = .1 + .1 * horizonofuncertainty#k is like a length scale
		return [sigma, alpha, k]
	end
	const nominalparams = params# + sqrt(params) .* randn(length(params)) / 100
	const ncompliancepoints = 10
	const ncompliancetimes = 10
	const compliancepoints = Array(Array{Float64, 1}, ncompliancepoints * ncompliancetimes)
	const compliancetimes = Array(Float64, ncompliancepoints * ncompliancetimes)
	for i = 1:ncompliancetimes
		for j = 1:ncompliancepoints
			compliancepoints[(i - 1) * ncompliancepoints + j] = [.5, (j - .5 * ncompliancepoints) / ncompliancepoints]
			compliancetimes[(i - 1) * ncompliancepoints + j] = 3. + i
		end
	end
	const compliancethreshold = 25.
	function performancegoalsatisfied(params::Vector, horizon::Number)
		results = model(params, compliancepoints, compliancetimes)
		results *= (1 + horizon)
		return !any(results .> compliancethreshold)
	end
	const x01bounds = [-1., 1.]
	const sigma01bounds = [1e-6, 1e-1]
	const v1bounds = [1e-2, 5e-1]
	const sigma1bounds = [sqrt(1e-2), sqrt(1e0)]
	const x02bounds = [-1., 1.]
	const sigma02bounds = [1e-6, 1e-1]
	const v2bounds = [-5e-1, 5e-1]
	const sigma2bounds = [sqrt(1e-4), sqrt(1e0)]
	const massbounds = [1e1, 3e2]
	const paramsmin = [x01bounds[1], sigma01bounds[1], v1bounds[1], sigma1bounds[1], x02bounds[1], sigma02bounds[1], v2bounds[1], sigma2bounds[1], massbounds[1]]
	const paramsmax = [x01bounds[2], sigma01bounds[2], v1bounds[2], sigma1bounds[2], x02bounds[2], sigma02bounds[2], v2bounds[2], sigma2bounds[2], massbounds[2]]
	function logprior(params::Vector)
		if any(params .< paramsmin) || any(params .> paramsmax)
			return -Inf
		else
			return 1.
		end
	end
	return BIGUQ.BigOED(model, data, xs, ts, proposedlocations, proposedtimes, makeresidualdistribution, residualdistributionparamsmin, residualdistributionparamsmax, nominalparams, performancegoalsatisfied, logprior)
end

bigoed = makebigoed1()
bigdt = BIGUQ.makebigdt(bigoed)
@time maxfailureprobs, horizons, badlikelihoodparams = BIGUQ.getrobustnesscurve(bigdt, 1., 19; numhorizons = 101)
BIGUQ.printresults(maxfailureprobs, horizons, badlikelihoodparams)
