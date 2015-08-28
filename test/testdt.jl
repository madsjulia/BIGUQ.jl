@everywhere import BIGUQ

function getbiguq1()
	function model(params)
		k = params[1]
		return k * 2
	end
	function makeloglikelihood(likelihoodparams)
		N = likelihoodparams[1]
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
	biguq = BIGUQ.BigDT(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
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
	biguq = BIGUQ.BigDT(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
end

function test(biguq::BIGUQ.BigDT)
	numhorizons = 10
	@time maxfailureprobs, horizons, badlikelihoodparams = BIGUQ.getrobustnesscurve(biguq, 10, 10; numhorizons=numhorizons)
	for i = 1:numhorizons
		println(horizons[i], ": ", maxfailureprobs[i], " -- ", badlikelihoodparams[i])
	end
end

biguq1 = getbiguq1()
test(biguq1)
biguq2 = getbiguq2()
test(biguq2)
