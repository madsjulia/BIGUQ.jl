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
	# @show mfp1mc
	# @show blp1mc
	mfp1mc_good = [0.0, 0.1164, 0.2757, 0.3148, 0.3154, 0.3159]
	blp1mc_good = [2.0, 2.946231092024565, 6.772567162017035, 13.700022617908882, 13.700022617908882, 13.700022617908882]
	for i=1:6
		@Test.test isapprox(mfp1mc_good[i], mfp1mc[i], atol=1e-6)
		@Test.test isapprox(blp1mc_good[i], blp1mc[i], atol=1e-6)
	end

	mfp1lh, blp1lh = testlhmc(biguq1, [0.], [10.])
	# @show mfp1lh
	# @show blp1lh
	mfp1lh_good = [0.0, 0.5295173961840629, 0.7901494633363426, 0.7912, 0.7916, 0.7921]
	blp1lh_good = [2.0, 4.455447953456097, 9.968367622709813, 12.932745314323602, 16.599380007688225, 16.599380007688225]
	for i=1:6
		@Test.test isapprox(mfp1lh_good[i], mfp1lh[i], atol=1e-6)
		@Test.test isapprox(blp1lh_good[i], blp1lh[i], atol=1e-6)
	end

	biguq2 = getbiguq2()
	mfp2mc, blp2mc = testmcmc(biguq2)
	# @show mfp2mc
	# @show blp2mc
	mfp2mc_good = [0.0, 0.0, 0.0, 0.0001, 0.1515, 0.2649]
	blp2mc_good = [4.0, 4.0, 4.0, -1.4189513935792624, -3.6115024767738935, -4.072071947615175]
	for i=1:6
		@Test.test isapprox(mfp2mc_good[i], mfp2mc[i], atol=1e-6)
		@Test.test isapprox(blp2mc_good[i], blp2mc[i], atol=1e-6)
	end

	mfp2lh, blp2lh = testlhmc(biguq2, [0.], [10.])
	# @show mfp2lh
	# @show blp2lh
	mfp2lh_good = [3.2203563160248066e-5, 3.227899734877979e-5, 4.5381718321358685e-5, 7.280440999805115e-5, 0.0038342906616330203, 0.9919609042270471]
	blp2lh_good = [4.0, 3.8629301185331304, 0.8641614468987395, -0.006790283263419461, -2.0757241384864766, -4.511496382242316]
	for i=1:6
		@Test.test isapprox(mfp2lh_good[i], mfp2lh[i], atol=1e-6)
		@Test.test isapprox(blp2lh_good[i], blp2lh[i], atol=1e-6)
	end
end

:passed