# Assess the convergence of the harmonics as the integration domain increases
import numpy as np
from IPython import embed
import pickle
import matplotlib
import matplotlib.pyplot as plt
import itertools

# Name of transducer
transducer_name = 'H102'
power = 100

# How many harmonics have been computed
n_harms = 5

# Set up plot
# Plotting convergence errors
matplotlib.rcParams.update({'font.size': 26})
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()
marker = itertools.cycle(('ko-', 'rs-', 'd-', 'x-', '*-'))

# filename = 'results/Pierre_H101_water_harmonic1.pickle'
# filename = 'results/H101_water_harmonic1.pickle'
filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_water_harmonic1.pickle'

with open(filename, 'rb') as f:
    bits = pickle.load(f)

line_1 = bits[0]
x_line = bits[1]

total = line_1
total_snip = np.zeros_like(total)
total_snip += total

for i_harm in range(0, n_harms - 1):
    # Load pickle file
    # filename = 'results/H101_water_harmonic' + str(i_harm+2) + '.pickle'
    # filename = 'results/Pierre_H101_water_harmonic' + str(i_harm+2) + '.pickle'
    filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_water_harmonic' + str(i_harm + 2) + '.pickle'

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
        if (rel_errors[i] < 1e-2):
        # if TOL[i] <2e-3:
            if (count == 0):
                count += 1
                print(i)
                print('HARMONIC ',i_harm+2)
                print('x coord of left edge of box:', xMinVals[i]) 
                print('y coord of base of box:', yMinVals[i])
                lammy = lam / (i_harm + 2)
                print('x dist in wavelengths: ', (roc - xMinVals[i]) / lam)
                print('y dist in wavelengths: ', (0 - yMinVals[i]) / lam)
                total_snip += line[i, :]


    plt.loglog(np.flip(TOL[:-1]), np.flip(rel_errors), next(marker), linewidth=2)
    # from IPython import embed;embed()

plt.grid(True)
# plt.ylim([7e-6, 1e0])
# plt.xlim([1e-2, 1e0])
# plt.xticks(np.array([1,2,3,4,5]))
# legend
# plt.legend(('Relative to $p_2$', 'Relative to $p_1+p_2$'),
#            shadow=True, loc=(0.7, 0.7), handlelength=1.5,
#            fontsize=20)

plt.xlabel(r'Relative magnitude cut-off for integral')
plt.ylabel(r'Relative $L^2$ error')
fig.savefig('results/conv.png')
plt.close()


# Plot harmonics
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()

for i_harm in range(0, n_harms):
    # Load pickle file
    # filename = 'results/Pierre_H101_water_harmonic' + str(i_harm+1) + '.pickle'
    # filename = 'results/H101_water_harmonic' + str(i_harm+2) + '.pickle'
    filename = 'results/' + transducer_name + '_power' + str(power) + \
            '_water_harmonic' + str(i_harm + 1) + '.pickle'

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
            '_water_harmonic' + str(i_harm + 2) + '.pickle'

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

