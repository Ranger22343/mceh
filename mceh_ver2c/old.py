def plot_schechter(p , cluster, thetitle=None, log=True, printp = False):
    m_s, phi_s, alpha = p
    if thetitle != None:
        plt.title(thetitle)
    if printp == True and thetitle == None:
        plt.title('m_s=%5.3f, phi_s=%5.3f, alpha=%5.3f' % tuple(p))
    plt.xlim(14,24)
    if log == True:
        plt.yscale('log')
    plt.xlabel('magnitude')
    plt.ylabel('number') 
    plt.stairs(cluster.observation, cluster.bins, label='observation', color = 'black')
    plt.stairs(cluster.bkg, cluster.bins, label='bkg', color = 'grey')
    sf = schechter_bins(m_s, phi_s, alpha)
    plt.ylim(bottom=0.01)
    plt.stairs(sf, cluster.bins, label='schetcher', color = 'blue')
    plt.stairs(cluster.bkg+sf, cluster.bins, label='bkg+schetcher', color = 'red')
    plt.legend()

def find_ini_args(cluster, plot_result = False, double = False, method = None):
    '''
    Find the proper initial values for MCMC Schechter fitting (not contain the randomization).
    Return the values (m_s1, phi_s1, alpha1, m_s2, phi_s2, alpha2), which the index 1 and 2 represent the lower magnitude part and the higher magnitude part seperately.
    '''
    if method is None:
        return [20.,20.,-1.]
    if double == True:
        low_hist, low_bin, high_hist, high_bin, median_bin_index = find_half(cluster)
        #width = low_bin[1]-low_bin[0]
        low_bins_center = low_bin[:-1] + np.diff(low_bin) / 2
        high_bins_center = high_bin[:-1] + np.diff(high_bin) / 2
        total_bins_center = cluster.bins[:-1] + np.diff(cluster.bins) / 2
        popt2, pcov2 = curve_fit(schechter_bin_m, high_bins_center, high_hist, bounds=([15,0.,-5], [25, 1000, 5.]),p0=[20.,20.,-1.])
        m_s2, phi_s2, alpha2 = popt2
        substracted_hist = (cluster.observation - cluster.bkg) - schechter_bins(m_s2, phi_s2, alpha2)
        popt1, pcov1 = curve_fit(schechter_bin_m, total_bins_center, substracted_hist, bounds=([15,0.,-5], [25, 1000, 5.]), p0=[20.,20.,-1.])
        if plot_result == True:
            plot_cluster = Cluster(cluster.index) if cluster.is_cut else cluster
            plt.stairs(schechter_bins(popt1[0],popt1[1],popt1[2])[:median_bin_index], low_bin, label = 'bright-end fitting')
            plt.stairs(schechter_bins(popt2[0],popt2[1],popt2[2])[median_bin_index-1:], high_bin, label = 'faint-end fitting')
            plt.stairs(plot_cluster.observation - plot_cluster.bkg, plot_cluster.bins, label='obs - bkg', color='black')
            plt.title('result of LS fitting')
            plt.xlabel('magnitude')
            plt.ylabel('number')
            plt.xlim(14,24)
            plt.legend()
        return np.append(popt1,popt2)
    else:
        interval = cluster.bins[1] - cluster.bins[0]
        thefitted = (cluster.observation - cluster.bkg)/interval
        bins_center = cluster.bins[:-1] + np.diff(cluster.bins) / 2
        try:
            popt, pcov = curve_fit(schechter, bins_center, thefitted, bounds=([15,0.,-5.], [25, 1000, 5.]), p0=[20.,20.,-1.])
        except:
            popt = [20.,20.,-1]
        print('popt=',popt)
        if plot_result == True:
            fit_result = schechter_bins(popt[0],popt[1],popt[2])
            plot_cluster = Cluster(cluster.index) if cluster.is_cut else cluster
            plt.stairs(fit_result, plot_cluster.bins, label = 'fitting', color = 'red')
            plt.stairs(plot_cluster.observation, plot_cluster.bins, label='observation', color='black')
            plt.stairs(plot_cluster.bkg, plot_cluster.bins, label='background', color='grey')
            plt.stairs(plot_cluster.bkg+fit_result, plot_cluster.bins, label='background+fitting', color='blue') 
            plt.title(f'Result of LS fitting for {ordinal(plot_cluster.index+1)} cluster')
            plt.xlabel('magnitude')
            plt.ylabel('number')
            plt.xlim(14,24)
            plt.legend()
        return popt
