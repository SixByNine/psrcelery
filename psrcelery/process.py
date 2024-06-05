#!/usr/bin/env python
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import psrcelery
import numpy as np
import multiprocessing
from scipy.optimize import minimize, basinhopping


def init_thread(celery):
    global _celery
    _celery = celery
    _celery.unpack_gp()
    return

def log_like(p):
    global _celery
    return _celery.log_likelihood(p)
def neg_log_like_thread(params):
    global _celery
    gp = _celery.cel_gp
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(_celery.y)


if __name__ == '__main__':
    # Parse arguments to read in input filename, min and max lengthscale and min width
    parser = argparse.ArgumentParser(description="Run the GP search for pulsar signals")
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('--min_length', type=float,default=40, help='Minimum lengthscale')
    parser.add_argument('--max_length', type=float,default=4000, help='Maximum lengthscale')
    parser.add_argument('--min_width', type=float,default=0.16, help='Minimum width')
    parser.add_argument('--nthread', type=int, default=4, help='Number of threads')
    parser.add_argument('--matern', action='store_true', help='Use Matern kernel')
    parser.add_argument('--output', type=str, default='celery_out.npz', help='Output file')
    parser.add_argument('--basinhop', action='store_true', help='Use basinhopping')
    parser.add_argument('--emcee', action='store_true', help='Use emcee')
    parser.add_argument('--nwalkers', type=int, default=8, help='Number of walkers')
    parser.add_argument('--niter', type=int, default=1200, help='Number of iterations')
    parser.add_argument('--thin', type=int, default=20, help='Thin the chain')
    parser.add_argument('--nburn', type=int, default=600, help='Burn in')
    parser.add_argument('--de', action='store_true', help='Use differential evolution')
    args = parser.parse_args()

    celery = psrcelery.celery.loadCelery(args.input)
    celery.use_standard_kernel(log_min_width=np.log10(args.min_width), min_length=args.min_length,
                               max_length=args.max_length,matern=args.matern)



    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    print("Solving GP")
    initial_params = celery.cel_gp.get_parameter_vector()
    bounds = celery.cel_gp.get_parameter_bounds()

    if args.emcee:
        import emcee, corner
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        ndim = len(celery.cel_gp.get_parameter_vector())
        nwalkers = args.nwalkers
        thin=args.thin
        nsamples=args.niter
        nburn = args.nburn
        p0 = celery.sample_uniform(nwalkers)

        if args.nthread > 1:
            celery.pack_gp()  # Pack the GP for pickling
            with multiprocessing.Pool(args.nthread, initializer=init_thread,initargs=(celery,)) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_like,pool=pool)
                state = sampler.run_mcmc(p0, nsamples, progress=True, store=True)
                chain = sampler.get_chain(flat=True, thin=thin, discard=nburn)

            celery.unpack_gp()
        else:
            init_thread(celery)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_like)
            state = sampler.run_mcmc(p0, nsamples, progress=True, store=True)
            chain = sampler.get_chain(flat=True, thin=thin, discard=nburn)
        log_medp = np.median((chain), axis=0)
        print(log_medp)
        corner.corner(chain, labels=celery.cel_gp.get_parameter_names(), truths=log_medp)
        plt.savefig(f"{args.output}.corner.pdf",bbox_inches='tight')
        celery.set_parameter_vector(log_medp)
    elif args.de:
        from scipy.optimize import differential_evolution
        def print_fun(x, convergence):
            print(f"{x} at converge {convergence:.4f}")

        if args.nthread > 1:
            celery.pack_gp()
            with multiprocessing.Pool(args.nthread, initializer=init_thread,initargs=(celery,)) as pool:
                init_thread(celery)
                soln = differential_evolution(neg_log_like_thread, bounds,callback=print_fun,workers=pool.map,updating='deferred')
                celery.unpack_gp()
        else:
            soln = differential_evolution(neg_log_like, bounds, args=(celery.y, celery.cel_gp))
        celery.set_parameter_vector(soln.x)
    elif args.basinhop:
        def print_fun(x, f, accepted):
            print(f"{x} at LL {-f:.2f}: accepted {accepted}")
        soln = basinhopping(neg_log_like, initial_params, niter=20, minimizer_kwargs={"bounds": bounds, "args": (celery.y, celery.cel_gp)},callback=print_fun)
        celery.set_parameter_vector(soln.x)
    else:
        soln = minimize(neg_log_like, initial_params,
                    method="L-BFGS-B", bounds=bounds, args=(celery.y, celery.cel_gp))

        celery.set_parameter_vector(soln.x)

    print(celery.parameter_vector)
    print(np.exp(celery.parameter_vector[1]), 10 ** celery.parameter_vector[2])

    print("Predicting Profiles")
    celery.predict_profiles_blockwise(nthread=args.nthread)

    ndays = celery.mjd[-1] - celery.mjd[0]
    nresamp = max(int(ndays / 10),1024)
    print(f"Predict Resampled days={ndays} resamp={nresamp}")
    celery.predict_profiles_resampled_blockwise(nresamp, nthread=args.nthread)
    print("Saving")
    celery.save(args.output)