import MCMC
import Wells

times = linspace(1, 30, 30)
deltaheads = zeros(30)
Qw = .1 #m^3/sec
K1 = 1e-3 #m/sec -- pervious
K2 = 1e-5 #m/sec -- semi-pervious
L1 = 100 #m
L2 = 200 #m
Sc1 = 7e-5 #m^-1 -- dense, sandy gravel
Sc2 = 1e-5 #m^-1 -- fissured rock
ra = .1 #m
R = 100 #m
omega = 1e3 #no resistance
deltah = 0 #m
r1 = 50 #m
function loglikelihood(params)
	Qw = params[1]
	K1 = params[2]
	K2 = params[3]
	L1 = params[4]
	L2 = params[5]
	Sc1 = params[6]
	Sc2 = params[7]
	ra = params[8]
	R = params[9]
	omega = params[10]
	deltah = params[11]
	r1 = params[12]
	if K1 < 0 || K2 < 0 || L1 < 0 || L2 < 0 || Sc1 < 0 || Sc2 < 0 || ra < 0 || R < 0 || omega < 0 || r1 < 0
		return -Inf
	end
	avcideltaheads = map(t -> Wells.avcideltahead(Qw, K1, K2, L1, L2, Sc1, Sc2, ra, R, omega, deltah, r1, t), times)
	v = deltaheads - avcideltaheads
	retval = -dot(v, v)
	return retval
end
#mymodel = MCMC.model(v -> -dot(v, v), grad=v -> -2 * v, init=ones(3))
#mychain = run(mymodel, MCMC.HMC(0.1), MCMC.SerialMC(steps=100000, burnin=10000))
params0 = [Qw, K1, K2, L1, L2, Sc1, Sc2, ra, R, omega, deltah, r1]
#mymodel = MCMC.model(v -> -dot(v, v), init=params0)
mymodel = MCMC.model(loglikelihood, init=params0)
rmw = MCMC.RWM(0.1)
smc = MCMC.SerialMC(steps=50000, burnin=5000)
mychain = MCMC.run(mymodel, rmw, smc)
MCMC.describe(mychain)
