# Assess the convergence of the harmonics as the integration domain increases
import numpy as np
from IPython import embed
import pickle
import matplotlib
import matplotlib.pyplot as plt
import itertools

# Name of transducer
transducer_name = 'H101'
power = 100
material = 'water'

# How many harmonics have been computed
n_harms = 5
nPerLam = 10

# Set up plot
# Plotting convergence errors
matplotlib.rcParams.update({'font.size': 24})
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(10, 7))
# fig = plt.figure()
ax = fig.gca()
marker = itertools.cycle(('ro-', 'bs-', 'gd-', 'mp-'))
# marker = itertools.cycle(('ko-','ko--', 'bs-','bs--', 'rd-','rd--',
#                           'mp-','mp--'))

# filename = 'results/Pierre_H101_water_harmonic1.pickle'
# filename = 'results/H101_water_harmonic1.pickle'
# filename = 'results/' + transducer_name + '_power' + str(power) + \
#             '_' + material + '_harmonic1.pickle'
filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_' + material + '_harmonic1_nPerLam' + str(nPerLam) + '.pickle'

with open(filename, 'rb') as f:
    bits = pickle.load(f)

line_1 = bits[0]
x_line = bits[1]

total = line_1
total_snip = np.zeros_like(total)
total_snip += total

norms = [np.linalg.norm(line_1)]

for i_harm in range(0, n_harms - 1):
    # Load pickle file
    # filename = 'results/H101_water_harmonic' + str(i_harm+2) + '.pickle'
    # filename = 'results/Pierre_H101_water_harmonic' + str(i_harm+2) + '.pickle'
    # filename = 'results/' + transducer_name + '_power' + str(power) + \
    #         '_' + material + '_harmonic' + str(i_harm + 2) + '.pickle'
    filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_' + material + '_harmonic' + str(i_harm + 2) + '_nPerLam' + str(nPerLam) + '.pickle'

    with open(filename, 'rb') as f:
        VARS = pickle.load(f)

    # Field along the central axis line
    line = VARS[0]
    

    # Add to total field
    total += line[-1, :]

    # Array of the tolerances considered in the convergence experiment
    TOL = VARS[1]

    xMinVals = VARS[2]
    xMaxVals = VARS[3]
    yMinVals = VARS[4]
    yMaxVals = VARS[5]
    roc = VARS[6]

    print('ROC = ', roc)
    k1 = VARS[7]
    lam = 2 * np.pi / k1

    # Preallocate array of relative errors to be computed
    rel_errors = np.zeros(line.shape[0]-1)

    # Distances
    WX = np.zeros(line.shape[0]-1)
    WY = np.zeros(line.shape[0]-1)

    # Compute errors
    count = 0
    for i in range(line.shape[0]-1):
        rel_errors[i] = np.linalg.norm(line[-1, :]-line[i, :]) / \
                        np.linalg.norm(line_1)
        # rel_errors[i] = np.linalg.norm(line[-1, :]-line[i, :]) / \
        #                 np.linalg.norm(line[-1, :])
        # rel_errors[i] = np.abs(np.max(np.abs(line[-1, :]))-
        #                        np.max(np.abs(line[i, :]))) / \
        #                 np.max(np.abs(line[-1, :]))

        WX[i] = (roc - xMinVals[i]) / (roc - xMinVals[-1])
        WY[i] = yMinVals[i] / yMinVals[-1]

        if (rel_errors[i] < 1e-2):
        # if TOL[i] <2e-3:
            if (count == 0):
                count += 1
                print('HARMONIC ',i_harm+2)
                print('x coord of left edge of box:', xMinVals[i]) 
                print('y coord of base of box:', yMinVals[i])
                lammy = lam / (i_harm + 2)
                print('x dist in wavelengths: ', (roc - xMinVals[i]) / lam)
                print('y dist in wavelengths: ', (0 - yMinVals[i]) / lam)
                total_snip += line[i, :]
                norms.append(np.linalg.norm(line[i, :]))
                print('Fraction of y domain = ', WY[i])
                print('Fraction of x domain = ', WX[i])
    # plt.semilogy(np.flip(WX), 100*np.flip(rel_errors), next(marker), linewidth=2)
    plt.semilogy(np.flip(WY), 100*np.flip(rel_errors), next(marker), linewidth=2)
    # plt.loglog(np.flip(TOL[:-1]), np.flip(rel_errors)*100, next(marker), linewidth=2)
    # from IPython import embed;embed()

plt.grid(True)

# plt.loglog(np.flip(TOL[:-1]), norms[2]/norms[0]*100*(np.flip(TOL[:-1]))**0.5, 'k--', linewidth=2)
# plt.semilogy(np.flip(WX), 1e-4*np.flip(WX)**(-2), 'k--', linewidth=2)
# plt.ylim([7e-6, 1e0])
# plt.yticks([1e-3, 1e-2, 1e-1,1e0,1e1], ('0.001','0.01','0.1','1','10'))
plt.xlim([0, 1.01])
# plt.xticks(np.array([1,2,3,4,5]))

# plt.text(3e-4, 4e-1, r'$10\frac{|p_i|}{|p_1|}\sqrt{Q_0}$',
#          {'color': 'k', 'fontsize': 20})

# legend
# plt.legend((r'$p_2$', r'$p_3$',r'$p_4$',r'$p_5$'),
#            shadow=False, loc=(0.84, 0.03), handlelength=1.5, fontsize=20)

plt.legend((r'$p_2$', r'$p_3$',r'$p_4$',r'$p_5$'),
           shadow=False, loc=(0.03, 0.03), handlelength=1.5, fontsize=20)

# plt.xlabel(r'$Q_0$')
plt.xlabel(r'Fraction of reference domain ($y/z$)')
plt.ylabel('Error (\%)')

filename = 'results/domain_convergence_space_y_' + material + transducer_name + '_power' + str(power) + '.pdf'
fig.savefig(filename)
plt.close()


# Plot harmonics
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(9, 7))
ax = fig.gca()

for i_harm in range(0, n_harms):
    # Load pickle file
    # filename = 'results/Pierre_H101_water_harmonic' + str(i_harm+1) + '.pickle'
    # filename = 'results/H101_water_harmonic' + str(i_harm+2) + '.pickle'
    filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_' + material + '_harmonic' + str(i_harm + 1) + '_nPerLam' + str(nPerLam) +'.pickle'

    with open(filename, 'rb') as f:
        VARS = pickle.load(f)


    # Field along the central axis line
    line = VARS[0]
    # from IPython import embed; embed()
    if (i_harm == 0):
        plt.plot(x_line, np.abs(line))
    else:
        plt.plot(x_line, np.abs(line[-1, :]))
    # plt.plot(x_line, np.abs(line[10, :]))
    plt.grid(True)

fig.savefig('results/harms.png')
plt.close()

# Plot harmonics
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()

for i_harm in range(0, 1):
    # Load pickle file
    # filename = 'results/Pierre_H101_water_harmonic' + str(i_harm+2) + '.pickle'
    # filename = 'results/H101_water_harmonic' + str(i_harm+2) + '.pickle'
    filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_' + material + '_harmonic' + str(i_harm + 2) + '_nPerLam' + str(nPerLam) +'.pickle'

    with open(filename, 'rb') as f:
        VARS = pickle.load(f)


    # Field along the central axis line
    line = VARS[0]

    plt.plot(np.abs(line[-10:-1, :]).T)

fig.savefig('results/harm_conv.png')
plt.close()


# Plot total field
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()

# plt.plot(x_line, np.real(total_snip))
plt.plot(x_line[600:], np.real(total[600:]))
fig.savefig('results/total.png')
plt.close()

# Compute error of total_snip
error_total = np.linalg.norm(total - total_snip) / np.linalg.norm(total)
print('Relative error of truncated domain approx = ', error_total)

