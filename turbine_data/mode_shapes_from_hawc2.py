import numpy as np
import matplotlib.pyplot as plt

#%% Read Hawc2 blade mode shapes file

n_modes = 6

amp_file = 'amplt100.amp'
amp = np.loadtxt(amp_file)

freq_file = 'freq_27.cmb'
freq = np.loadtxt(freq_file)
omega = freq[1:n_modes+1]*(2.*np.pi)

phi = np.zeros((len(amp[:, 1]), 1+(3*9)))
phi[:, 0] = amp[:, 1]+2.8
for i_m in range(n_modes):
    for i_d in range(3):
        col_0 = int(6*i_m + 2*i_d + 2)
        col_1 = int(3*i_m + 1*i_d + 1)
        phi[:, col_1] = amp[:, col_0]*np.cos(amp[:, col_0+1]*(np.pi/180.))

#%% Print

for i_m in range(n_modes):
    # i_m = 0
    fig = plt.figure()
    plt.plot(phi[:, 0], phi[:, 3*i_m+0+1], 'C0', label=r'$phi_x$')
    plt.plot(phi[:, 0], phi[:, 3*i_m+1+1], 'C1', label=r'$phi_1$')
    plt.xlabel(r'z [m]')
    plt.ylabel(r'$\phi$ [-]')
    plt.title(r'mode %i'%(i_m+1))
    plt.legend()
    plt.grid('on')

#%% Save file

# Mode shapes
np.savetxt('mode_shapes.dat', phi, fmt='%0.10e', delimiter='\t', newline='\n', header='# mode shapes', footer='', comments='# ', encoding=None)

# Angular frequencies
np.savetxt('mode_frequencies.dat', omega, fmt='%0.10e', delimiter='\t', newline='\n', header='# mode angular frequencies [rad/s]', footer='', comments='# ', encoding=None)


