dataN = np.loadtxt("corr_N.dat")
dataR = np.loadtxt("corr_R.dat")
assert np.all(dataN[:, 1] == dataR[:, 1])
r_ft = dataN[:, 1]
xi_0_ft = dataN[:, 3] / dataR[:, 3]
xi_2_ft = dataN[:, 4] / dataR[:, 4]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_ft, xi_0_ft*np.square(r_ft), 'g.--', ms=8)
fig.savefig('xi_0.pdf')
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_ft, xi_2_ft*np.square(r_ft), 'g.--', ms=8)
fig.savefig('xi_2.pdf')