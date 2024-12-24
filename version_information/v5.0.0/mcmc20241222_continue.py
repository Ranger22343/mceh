import mceh.model.all_zm_model_schechter as fitting
import mceh.utility as ut
import multiprocessing

multiprocessing.set_start_method('fork', force=True)

cpu_num = 40
cpu4fit = 32
ut.printt('code name: mcmc20241222_continue.py')
ut.printt(f'cpu: {cpu_num}')
ut.printt('memory: 32G')
ut.printt('Start')
previous_result = ut.pickle_load('result/mcmc20241220_continue.pickle')
sampler = previous_result['sampler']
state = previous_result['state']
is_used = previous_result['is_used']
mcmc_dict = previous_result['mcmc_dict']
ut.printt('Start MCMC')
with multiprocessing.Pool(cpu4fit) as pool:
    sampler.pool = pool
    state = sampler.run_mcmc(state, 10000, progress=True)
result = {'sampler': sampler, 'state': state, 'is_used': is_used,
          'mcmc_dict': mcmc_dict}
ut.pickle_dump(result, 'result/mcmc20241222_continue.pickle')
ut.printt('All Finish')