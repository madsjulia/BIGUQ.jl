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

@Test.testset "BIGUQ" begin
	biguq1 = getbiguq1()
	mfp1mc, blp1mc = testmcmc(biguq1)
	mfp1mc_good = [0.0, 0.2556, 0.2822, 0.2994, 0.3482, 0.3484]
	blp1mc_good = [2.0, 5.2844070360185285, 7.444968205794697, 13.556910220654995, 15.745641078412042, 15.745641078412042]
	for i=1:6
		@Test.test isapprox(mfp1mc_good[i], mfp1mc[i], atol=1e-6)
		@Test.test isapprox(blp1mc_good[i], blp1mc[i], atol=1e-6)
	end

	mfp1lh, blp1lh = testlhmc(biguq1, [0.], [10.])
	mfp1lh_good = [0.0, 0.3619482496194825, 0.7874351936566026, 0.7913, 0.7917, 0.7921]
	blp1lh_good = [2.0, 3.2846472088725, 9.837369124373081, 11.616288946429627, 11.616288946429627, 11.616288946429627]
	for i=1:6
		@Test.test isapprox(mfp1lh_good[i], mfp1lh[i], atol=1e-6)
		@Test.test isapprox(blp1lh_good[i], blp1lh[i], atol=1e-6)
	end

	biguq2 = getbiguq2()
	mfp2mc, blp2mc = testmcmc(biguq2)
	mfp2mc_good = [0.0, 0.0, 0.0, 0.0, 0.0434, 0.0788]
	blp2mc_good = [4.0, 4.0, 4.0, 4.0, -3.761337647310173, -4.002622914064428]
	for i=1:6
		@Test.test isapprox(mfp2mc_good[i], mfp2mc[i], atol=1e-6)
		@Test.test isapprox(blp2mc_good[i], blp2mc[i], atol=1e-6)
	end

	mfp2lh, blp2lh = testlhmc(biguq2, [0.], [10.])
	mfp2lh_good = [3.207380832206222e-5, 3.216220137003713e-5, 3.56774862493778e-5, 0.00013339205587499655, 0.0019817679913793655, 0.998522362632795]
	blp2lh_good = [4.0, 3.7846025434923813, 1.6603780645678023, -0.9524196631197999, -2.291791981482562, -5.8479072462287505]
	for i=1:6
		@Test.test isapprox(mfp2lh_good[i], mfp2lh[i], atol=1e-6)
		@Test.test isapprox(blp2lh_good[i], blp2lh[i], atol=1e-6)
	end
end

:passed