import mceh
import mceh.model.group_redblue as fitting
import numpy as np


cpu_num = 8


efeds, hsc = fitting.init('sr_efeds', 'hsc')
rs_rd_result = fitting.init('rs_rd_result')
rs_data = fitting.init('rs_data')
index_group = fitting.efeds_index_group(efeds)
result = {'all_mcmc_dict': [], 'all_sampler': [], 'all_index': index_group}


for i in range(len(index_group)):
    mceh.printt(f'Processing {i + 1}/{len(index_group)}')
    mcmc_dict = fitting.easy_mcmc(index_group[i],
                                  efeds,
                                  hsc,
                                  rs_rd_result,
                                  rs_data,
                                  mode='red')
    result['all_mcmc_dict'].append(mcmc_dict)
    sampler = fitting.get_sampler(**mcmc_dict, progress=True, cpu_num=cpu_num)
    result['all_sampler'].append(sampler)
mceh.pickle_dump(result, 'result/20250607mcmc_red.pickle')
mceh.printt('Finished!')
