import mceh.model.all_zm_model_schechter as fitting
import mceh.utility as ut

ut.printt('code name: mcmc20241219.py')
ut.printt('cpu: 32')
ut.printt('memory: 32G')
ut.printt('Start')
efeds, hsc = ut.init('efeds', 'hsc')
rd_result = ut.pickle_load('result/bkg_lf20241111.pickle')

efeds = efeds[efeds['low_cont_flag'] & (efeds['unmasked_fraction'] > 0.6)]
ut.printt('Start MCMC')
mcmc_dict = fitting.easy_mcmc(list(range(len(efeds))), efeds, hsc, rd_result)
sampler, state, is_used = fitting.get_sampler(progress=True, cpu_num=32,
                                              step=10000, 
                                              **mcmc_dict)
result = {'sampler': sampler, 'state': state, 'is_used': is_used,
          'mcmc_dict': mcmc_dict}
ut.pickle_dump(result, 'result/mcmc20241219.pickle')
ut.printt('All Finish')