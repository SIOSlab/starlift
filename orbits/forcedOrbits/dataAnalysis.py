import numpy as np
import sys
import os
import astropy.units as u
import astropy.constants as const
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})

fileDirectory = '/Users/gracegenszler/Documents/Research/starlift/orbits/forcedOrbits/'
orbitName = 'TrajI_1265/'
#thrusterName = 'ST-100 Hall Thruster'
#thrusterName = ''
#thrusterName = 'BHT-1500 Hall Thruster'
#paramName = 'Thrust'
paramName = 'Isp'
#paramName = 'Mass'

fileNames = np.array([paramName,'Thrust','Isp'])
thrusterNames = np.array(['ST-100 Hall Thruster', 'BHT-1500 Hall Thruster', 'BHT-1500 Hall Thruster'])
labels = np.array(['ST-100 Hall Thruster', 'BHT-1500 Hall Thruster: High Thrust Mode', 'BHT-1500 Hall Thruster: High Specific Impulse Mode'])
lineStyles = np.array(['b', 'g-.', 'y--'])
for jj in np.arange(len(thrusterNames)):
    Fts = np.array([])
    Isps = np.array([])
    t_burns = np.array([])
    pos_errors = np.array([])
    mis = np.array([])
    mfs = np.array([])
    thrusterName = thrusterNames[jj]
    fileName = fileNames[jj]
    with open(fileDirectory+orbitName+thrusterName+'_vary'+fileName+'.txt', 'r') as ff:
        lines = ff.readlines()
        for ii in np.arange(1,len(lines)):
            vals = lines[ii].split(', ')
            Fts = np.append(Fts, float(vals[0]))
            Isps = np.append(Isps, float(vals[1]))
            t_burns = np.append(t_burns, float(vals[2]))
            pos_errors = np.append(pos_errors, float(vals[3][:-1]))

            if paramName == 'Mass':
                mis = np.append(mis, float(vals[4]))
                mfs = np.append(mfs, float(vals[5][:-1]))
    
    Fts = Fts*u.mN
    Isps = Isps*u.s
    t_burns = t_burns*u.s
    pos_errors = pos_errors*u.km
    
    if paramName == 'Thrust':
        plt.figure(1, figsize=(16, 8))
        plt.figure(2, figsize=(16, 8))
                
    elif paramName == 'Mass':
        paramName2 = paramName
        m_per = (mis - mfs)/mis*100
        m_legend = np.array(["ST-25 Hall Thruster", "ST-40 Hall Thruster", "BHT-100 Hall Thruster", "BHT-200 Hall Thruster",  "BHT-350 Hall Thruster", "BHT-600 Hall Thruster", "BHT-6000 Hall Thruster" , "BHT-20k Hall Thruster", "Reference"])
        m_markers = np.array(["*", "o", "v", "s", "p", "d", "P", "X", "h"])
        m_color = np.array(["blue", "orange", "green", "red", "purple", "brown", "pink", "olive", "cyan"])
        m_inds = np.array([0, 2, 3, 0, 1, 4, 1, 5, 8, 6, 6, 7])
        
        plt.figure(5, figsize=(16, 8))
        plt.figure(6, figsize=(16, 8))
        plt.figure(7, figsize=(16, 8))
        plt.figure(8, figsize=(16, 8))
                    
        fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 8))
        fig7, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 8))
        
    else:
        plt.figure(3, figsize=(20, 8))
        plt.figure(4, figsize=(20, 8))
        
        ind_order = np.argsort(Isps)
        Isps_plt = np.sort(Isps)
        t_burns_plt = t_burns[ind_order]
        pos_errors_plt = pos_errors[ind_order]

    if paramName == 'Thrust':
        plt.figure(1, figsize=(16, 8))
        plt.yscale('log')
        plt.plot(Fts.value, t_burns, lineStyles[jj], label=labels[jj])
        plt.xlabel('Force [mN]')
        plt.ylabel('Total Burn Time [s]')
#        plt.title(thrusterName)
        plt.legend()
        plt.savefig(fileDirectory+orbitName+'thrustVsBurnTime.png')

        plt.figure(2, figsize=(16, 8))
        #plt.yscale('log')
        plt.plot(Fts.value, pos_errors.value, lineStyles[jj], label=labels[jj])
        plt.xlabel('Force [mN]')
        plt.ylabel('Max Position Error [km]')
#        plt.title(thrusterName)
        plt.legend()
        plt.savefig(fileDirectory+orbitName+'thrustVsPosError.png')
        
    elif paramName == 'Isp':
        plt.figure(3, figsize=(20, 8))
        plt.plot(Isps_plt.value, t_burns_plt, lineStyles[jj], label=labels[jj])
        plt.xlabel('Specific Impulse [s]')
        plt.ylabel('Total Burn Time [s]')
#        plt.title(thrusterName)
        plt.legend(loc='upper left')
        plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.11) 
        plt.savefig(fileDirectory+orbitName+'ispVsBurnTime.png')

        plt.figure(4, figsize=(20, 8))
        plt.plot(Isps_plt.value, pos_errors_plt.value, lineStyles[jj], label=labels[jj])
        plt.xlabel('Specific Impulse [s]')
        plt.ylabel('Max Position Error [km]')
#        plt.title(thrusterName)
        plt.legend(loc='upper left')
        plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.11)
        plt.savefig(fileDirectory+orbitName+'ispVsPosError.png')
        
    elif paramName == 'Mass':
        for ii in np.arange(len(m_per)):
            plt.figure(5, figsize=(16, 8))
            plt.scatter(Fts[ii].value, t_burns[ii].to_value(u.hr), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])

            plt.figure(6, figsize=(16, 8))
            plt.scatter(Fts[ii].value, pos_errors[ii].to_value(u.m), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])
            
            plt.figure(7, figsize=(16, 8))
            plt.scatter(Isps[ii].value, t_burns[ii].to_value(u.hr), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])

            plt.figure(8, figsize=(16, 8))
            plt.scatter(Isps[ii].value, pos_errors[ii].to_value(u.m), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])
            
            ax5.scatter(m_per[ii], t_burns[ii].to_value(u.hr), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])
            ax6.scatter(mis[ii]-mfs[ii], t_burns[ii].to_value(u.hr), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])

            ax7.scatter(m_per[ii], pos_errors[ii].to_value(u.m), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])
            ax8.scatter(mis[ii]-mfs[ii], pos_errors[ii].to_value(u.m), c=m_color[m_inds[ii]], marker = m_markers[m_inds[ii]], label=m_legend[m_inds[ii]])
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        ax5.set_xlabel('Fuel Mass Percent Used [%]')
        ax6.set_xlabel('Fuel Mass Used [kg]')
        ax5.set_ylabel('Total Burn Time [hr]')
        ax6.get_yaxis().set_visible(False)
        fig5.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(.9, .11))
        fig5.savefig(fileDirectory+orbitName+thrusterName+paramName2+'massVsBurnTime.png')

        ax7.set_xlabel('Fuel Mass Percent Used [%]')
        ax8.set_xlabel('Fuel Mass Used [kg]')
        ax7.set_ylabel('Max Position Error [m]')
        ax8.get_yaxis().set_visible(False)
        fig7.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(.9, .88))
        fig7.savefig(fileDirectory+orbitName+thrusterName+paramName2+'massVsPosError.png')

        plt.figure(5)
        plt.xscale('log')
        plt.xlabel('Force [mN]')
        plt.ylabel('Total Burn Time [hr]')
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(fileDirectory+orbitName+thrusterName+paramName2+'thrustVsBurnTime.png')

        plt.figure(6)
        plt.xscale('log')
        plt.xlabel('Force [mN]')
        plt.ylabel('Max Position Error [m]')
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(fileDirectory+orbitName+thrusterName+paramName2+'thrustVsPosError.png')
        
        plt.figure(7)
        plt.xlabel('Specific Impulse [s]')
        plt.ylabel('Total Burn Time [hr]')
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(fileDirectory+orbitName+thrusterName+paramName2+'ispVsBurnTime.png')

        plt.figure(8)
        plt.xlabel('Specific Impulse [s]')
        plt.ylabel('Max Position Error [m]')
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(fileDirectory+orbitName+thrusterName+paramName2+'ispVsPosError.png')

plt.show()
breakpoint()
