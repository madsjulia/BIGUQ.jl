import Test
import BIGUQ
import BlackBoxOptim
import Random

Random.seed!(0)

function getbiguq1()
	function model(params)
		k = params[1]
		return k * 2
	end
	function makeloglikelihood(likelihoodparams)
		N = likelihoodparams[1][1]
		return params -> (params[1][1] <= N ? 0. : -Inf)
	end
	function logprior(params)
		k = params[1][1]
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
	function model(params::AbstractVector)
		return params[1]
	end
	data = 1 .+ .1 .* randn(5)
	function makeloglikelihood(likelihoodparams::AbstractVector)
		logvar = likelihoodparams[1]
		var = exp.(logvar)
		return params -> -.5 * sum((data .- params[1]) .^ 2) / var - logvar
	end
	nominalparams = [.5]
	function logprior(params::Vector)
		return -.5 .* (params[1] - nominalparams[1]) .^ 2 / .01
	end
	function likelihoodparamsmin(horizon::Number)
		return [4 .- horizon]
	end
	function likelihoodparamsmax(horizon::Number)
		return [4 .+ horizon]
	end
	function performancegoalsatisfied(params::AbstractVector, horizon::Number)
		return model(params) < .9
	end
	biguq = BIGUQ.BigDT(makeloglikelihood, logprior, nominalparams, likelihoodparamsmin, likelihoodparamsmax, performancegoalsatisfied)
end

function testmcmc(biguq::BIGUQ.BigDT)
	numhorizons = 6
	@time maxfailureprobs, horizons, badlikelihoodparams=BIGUQ.getrobustnesscurve(biguq, 10, 10; numhorizons=numhorizons)
	badlikelihoodparamsv = map(p->p[1], badlikelihoodparams)
	for i = 1:numhorizons
		println(horizons[i], ": ", maxfailureprobs[i], " => ", badlikelihoodparamsv[i])
	end
	return maxfailureprobs, badlikelihoodparamsv
end

function testlhmc(biguq::BIGUQ.BigDT, modelparamsmin::AbstractVector, modelparamsmax::AbstractVector)
	numhorizons = 6
	modelparams = BlackBoxOptim.Utils.latin_hypercube_sampling(modelparamsmin, modelparamsmax, 10000)
	getfailureprobs = BIGUQ.makegetfailureprobabilities_mc(modelparams)
	maxfailureprobs, horizons, badlikelihoodparams=BIGUQ.getrobustnesscurve(biguq, 10, 10; getfailureprobfnct=getfailureprobs, numhorizons=numhorizons)
	badlikelihoodparamsv = map(p->p[1], badlikelihoodparams)
	for i = 1:numhorizons
		println(horizons[i], ": ", maxfailureprobs[i], " => ", badlikelihoodparamsv[i])
	end
	return maxfailureprobs, badlikelihoodparamsv
end

function testrobustnesscurve(maxfailureprobs::AbstractVector{<:Real}, badlikelihoodparams::AbstractVector{<:Real}, nominalparam::Real)::Nothing
	@Test.test length(maxfailureprobs) == 6
	@Test.test length(badlikelihoodparams) == 6
	@Test.test all(isfinite, maxfailureprobs)
	@Test.test all(isfinite, badlikelihoodparams)
	@Test.test all(p -> 0 <= p <= 1, maxfailureprobs)
	@Test.test all(diff(maxfailureprobs) .>= -eps(Float64))
	@Test.test isapprox(first(badlikelihoodparams), nominalparam; atol=1e-8)
	return nothing
end

@Test.testset "BIGUQ" begin
	biguq1 = getbiguq1()
	mfp1mc, blp1mc = testmcmc(biguq1)
	testrobustnesscurve(mfp1mc, blp1mc, 2.)
	@Test.test last(mfp1mc) > 0.1

	mfp1lh, blp1lh = testlhmc(biguq1, [0.], [10.])
	testrobustnesscurve(mfp1lh, blp1lh, 2.)
	@Test.test last(mfp1lh) > 0.5

	biguq2 = getbiguq2()
	mfp2mc, blp2mc = testmcmc(biguq2)
	testrobustnesscurve(mfp2mc, blp2mc, 4.)
	@Test.test last(mfp2mc) > 0.01

	mfp2lh, blp2lh = testlhmc(biguq2, [0.], [10.])
	testrobustnesscurve(mfp2lh, blp2lh, 4.)
	@Test.test last(mfp2lh) > 0.9
end

:passed
