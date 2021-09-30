import math as mt
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

""" (Robust) Compressed Sensing (CS) and harmonic signals (sum of sinusoids)


    Parameters
    ----------
    S : int
        Number of non-zeros coefficients of the S-sparse signal representation.
    pts : int
        Number of points of the original signal (sinusoid sum).
    n : int
        Number of FFT points and, consequently, of the sparse signal points.
    N : int
        Number of columns of the sensing matrix.
    M : int
        Number of rows of the sensing matrix.
    t : (pts,) ndarray
        Range (in rads) of values used by the reference sinusoid.
    f : (S,) ndarray
        Random frequency values with uniform distribution.
    A : (S,) list
        Random amplitude values with uniform distribution.
    Phi : (M,N) ndarray
        Sensing (aquisition) matrix generated from i.i.d Gaussian samples.
    wNoise : (M,) ndarray
        Sensing noise with (Normal) Gaussian distribution ~ N(0,1).
    s : (pts,) ndarray
        Superposition of S sinusoids with random frequencies/amplitudes.
    fft : (n,) ndarray
        DFT of the sum of sinusoids using FFT algorithm.
    x : (n,) ndarray
        S-sparse (compressible) signal.
    idx : (n,) ndarray
        Sorted index values of largest coefficients (in magnitude) in 'x'.
    y : (M,) ndarray
        Compressed signal.
    eps : float
        Second Order Cone Program (SOCP) constraint constant for robust CS.
    xOpt
        CVXPY optimization problem variable and sparsest solution.
    SOC_constraints : list
        CVXPY problem second order cone constraints for robust CS.
    prob
        CVXPY optimization probem formulation.
    sCs : (pts,) ndarray
        Signal (sum of sinusoids) reconstruction with CS.
    fs : int
        Sampling frequency for linear (Nyquist-Shannon) reconstruction.
    Psi : (pts,pts) ndarray
        Sensing (aquisition) identity matrix generated for linear sampling.
    uniIdx : (fs,) ndarray
        Indexes of the positions sampled from the original signal (Dirac comb).
    yNs : (fs,) ndarray
        Signal after linear sampling.
    sNs : (pts,) ndarray
        Signal (sum of sinusoids) reconstruction with sinc interpolation.


    Notes
    -----
    This code implements a (robust) CS scheme in which a sum of sinusoids with
    random amplitudes/frequencies is recovered using l_1 minimization. For
    comparison purposes, this simulation also includes the Nyquist-Shannon
    linear sampling scheme.


    References
    ----------

    .. [1] E. J. Candes and M. B. Wakin, "An Introduction To Compressive
           Sampling," in IEEE Signal Processing Magazine, vol. 25, no. 2,
           pp. 21-30, March 2008.

    .. [2] Han, Z., Li, H., & Yin, W. (2013). Compressive Sensing for Wireless
           Networks. Cambridge: Cambridge University.

    .. [3] Eldar, Y., & Kutyniok, G. (Eds.). (2012). Compressed Sensing:
           Theory and Applications. Cambridge: Cambridge University Press.

    .. [4] S. Boyd and L. Vandenberghe, Convex Optimization. New York, NY, USA:
           Cambridge University Press, 2004.


    © 2020 Pedro H. C. de Souza
"""
S = 16
pts = 2047
n = (pts + 1)//2
N = n  # the 'n'-point FFT sparsifies the signal, thus dim(Range('Phi')) = 'n'
M = 91  # m >= C*S*log(N/S) for i.i.d Gaussian sensing matrix (to obey RIP)
t = np.linspace(0, 2*np.pi, pts)  # ref. sinusoid period spans 'pts' samples
np.random.seed(1)  # repeat a run with the same aplitudes/frequecies values
f = np.random.randint(1, 100, S)  # location of the S non-zero coefficients
A = np.random.randint(5, 10, S)  # magnitude of the S non-zero coefficients
np.random.seed()  # a new realization of the sensing matrix for each run
Phi = np.random.randn(M, N)/mt.sqrt(M)  # ~ N(0,1/M) (variance is normalized)
wNoise = (np.random.randn(M) + 1j*np.random.randn(M))/mt.sqrt(2)  # ~ N(0,1)

s = np.zeros(pts)
for i in range(S):
    s += A[i]*np.sin(f[i]*t)  # generate the sum of sinusoids
fft = np.fft.rfft(s, norm="ortho")
# x = fft  # uncomment (comment below) to add signal noise

# Optimum thresholding strategy (see [3, pg.11])
x = np.zeros(N) + 1j*np.zeros(N)
idx = np.argsort(-np.abs(fft)**2)  # sort coefficients in descend order
x[idx[0:S]] = fft[idx[0:S]]  # oracle that selects the largest coefficients

# Universal encoding
y = Phi@x.T  # sensing (aquisition) of the sparse (compressible) signal
# y = Phi@x.T + wNoise  # uncomment (comment above) to add sensing noise

# l_1 minimization reconstruction
xOpt = cp.Variable(N, complex=True)
# eps = np.linalg.norm(wNoise)  # see [2, pg.64] Theorem 4
# SOC_constraints = [cp.SOC(cp.real(eps), Phi@xOpt.T - y)]  # uncomment
# prob = cp.Problem(cp.Minimize(cp.norm(xOpt, p=1)),        # (comment below)
#                   SOC_constraints)                        # for robust CS
prob = cp.Problem(cp.Minimize(cp.norm(xOpt, p=1)),
                  [Phi@xOpt.T == y])  # l_1 minization problem in CVXPY
prob.solve()
sCs = np.fft.irfft(xOpt.value, pts, norm="ortho")  # IFFT performed on "x*"

# Nyquist-Shannon linear sampling
fs = 161
Psi = np.identity(pts)
uniIdx = np.round(np.arange(0, pts - 1, pts/fs)).astype(int)
yNs = Psi[uniIdx, :]@s.T
sNs = np.zeros(pts)
for i in range(pts):
    sNs[i] = yNs @ np.sinc((t[i]/(2*np.pi) - np.arange(fs)*(1/fs))/(1/fs))

# Plot results
fig, ax = plt.subplots()
plt.subplot(211)
line1, = plt.plot(t, s, 'k', label='Original')
line2, = plt.plot(t, sNs, '--g', label='Nyquist-Shannon')
line3, = plt.plot(t, sCs, ':r', label='CS')
plt.axis([np.pi, np.pi + 0.5, np.min(s), np.max(s)])
plt.xlabel('samples')
plt.ylabel('Amplitude')
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.legend(fontsize='small')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = '\n'.join((
    'Nyquist-Shannon \n M = %.2i samples \n $|| \mathbf{s^*} - \mathbf{s^o} ||^2_2 /N$ = %.2E' %
    (fs, np.linalg.norm(sNs - s)**2 / s.size / (sum(A**2) / 2)),
    'CS \n M = %.2i samples \n $|| \mathbf{s^*} - \mathbf{s^o} ||^2_2 /N$ = %.2E' %
    (M, np.linalg.norm(sCs - s)**2 / s.size / (sum(A**2) / 2))))
plt.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize='xx-small',
         verticalalignment='top', bbox=props)

plt.subplot(212)
normCf = (np.linalg.norm(s)**2)/2
line1, = plt.plot(np.arange(n), (np.abs(x)**2)/normCf,
                  'xb', markersize=4, label='Original (FFT)')
line2, = plt.plot(np.arange(n), (np.abs(xOpt.value)**2)/normCf, 'or', fillstyle='none',
                  markersize=4, label='CS')
plt.axis([0, np.max(f) + 20, 0, 0.15])
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.yticks(visible=False)
plt.legend(fontsize='small')
textstr = r'$S = %i$' % (S, )
plt.text(0.9, 0.05, textstr, transform=ax.transAxes, fontsize='small',
         verticalalignment='bottom', bbox=props)
