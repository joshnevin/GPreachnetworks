# Implementation of most likely heteroscedastic GP regression algroithm:
# K. Kersting, C. Plagemann, P. Pfaff, and W. Burgard,
# “Most likely heteroscedastic gaussian process regression,” 
# inProc. 24th InternationalConference on Machine Learning,(Oregon, USA, 2007), pp. 393–400.

# %%  imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import time
from numpy.linalg import cholesky
from numpy import transpose as T
from numpy.linalg import inv, det, solve
from numpy import matmul as mul, exp
from numpy.random import normal
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import norm
#import multiprocessing
#from GHquad import GHquad
#import matplotlib
import math
#matplotlib.rc_file_defaults()   # use to return to Matplotlib defaults 

snr = np.genfromtxt(open("hetdata20.csv", "r"), delimiter=",", dtype =float) # run heteroscedastic datagen section from GPreachringsrand.py 
#snr = np.genfromtxt(open("hetdataextdegdv.csv", "r"), delimiter=",", dtype =float) # run heteroscedastic datagen section from GPreachringsrand.py 
#snr = np.genfromtxt(open("hetdataextdegst.csv", "r"), delimiter=",", dtype =float) # run heteroscedastic datagen section from GPreachringsrand.py 
numpoints = np.size(snr,1)
numedges = np.size(snr,0)
#x = np.linspace(0,numpoints-1,numpoints)
x = np.linspace(0,10,numpoints)
# 
#SNR = SNR[0]
#snr = snr[0:1]
# # data used by Goldberg - use for testing 
# =============================================================================
""" numpoints = 100
numedges = 1
x = np.linspace(0,1,numpoints).reshape(numpoints,numedges)
sd = np.linspace(0.5,1.5,numpoints)
y = np.zeros(numpoints)
n = np.size(x)
for i in range(numpoints):
    y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd[i])
snr = y.reshape(numpoints,numedges) """
    # =============================================================================
    
    # data used by Yuan and Wahba - use for testing 
    # =============================================================================
""" numpoints = 200
x = np.linspace(0,1,numpoints)  
ymean = 2*(exp(-30*(x-0.25)**2) + np.sin(np.pi*x**2)) - 2
sd = exp(np.sin(2*np.pi*x))
y = np.random.normal(ymean, sd)
snr = y """
    # =============================================================================
    
    # data used by Williams - use for testing 
""" numpoints = 200
x = np.linspace(0,np.pi,numpoints)
wmean = np.sin(2.5*x)*np.sin(1.5*x)
sd = 0.01 + 0.25*(1 - np.sin(2.5*x))**2
y = np.random.normal(wmean, sd)
snr = y """

def HGPfunc(x,y,plot):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    if plot:
        plt.plot(x,y,'+')
        plt.xlabel("Pch (dBm)")
        plt.ylabel("SNR (dB)")
        plt.savefig('Adataset.png', dpi=200)
        plt.show()
    n = np.size(x)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    
    def sqexp(X,Y,k1,k2):
        X = np.atleast_2d(X)
        if Y is None:
            dists = pdist(X / k2, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
            # return gradient 
            K_gradient = (K * squareform(dists))[:, :, np.newaxis]
            #K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \  # anisotropic case, see https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/gaussian_process/kernels.py
            #            / (k2 ** 2)
            #K_gradient *= K[..., np.newaxis]
            return k1*K, K_gradient
        else:
            dists = cdist(X / k2, Y / k2,metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            return k1*K
    # heteroscedastic versions of functions 
    global Kyinvh
    Kyinvh = 0.0
    global Kfh
    Kfh =  0.0 
    def lmlh(params,y,R):
        #print(params)  # show progress of fit
        [k1, k2] = params
        global Kfh
        Kfh = sqexp(x,None,k1,k2**0.5)[0]
        Ky = Kfh + R # calculate initial kernel with noise
        global Kyinvh
        Kyinvh = inv(Ky)
        return -(-0.5*mul(mul(T(y),Kyinvh), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)
    def lmlgh(params,y,R):
        k1, k2 = params
        al = mul(Kyinvh,y)
        dKdk1 = Kfh*(1/k1)
        dKdk2 = sqexp(x,None,k1,k2**0.5)[1].reshape(n,n)
        lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk1)))
        lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk2)))
        return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)
    def GPRfith(xs,k1,k2,R,Rs):
        Ky = sqexp(x,None,k1,k2**0.5)[0] + R
        Ks = sqexp(xs, x, k1, k2**0.5)
        Kss = sqexp(xs, None, k1, k2)[0]
        L = cholesky(Ky)
        al = solve(T(L), solve(L,y))
        fmst = mul(Ks,al)
        varfmst = np.empty([n,1])
        for i in range(np.size(xs)):
            v = solve(L,T(Ks[:,i]))
            varfmst[i] = Kss[i,i] - mul(T(v),v)  + Rs[i,i]
        lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
        #return fmst, varfmst[::-1], lmlopt
        return fmst, varfmst, lmlopt
    def hypopth(y, numrestarts, R):
        numh = 2 # number of hyperparameters in kernel function 
        k1s4 = np.empty([numrestarts,1])
        k2s4 = np.empty([numrestarts,1])
        for i in range(numrestarts):    
            #k1is4 = np.random.uniform(1e-2,1e3)
            #k2is4 = np.random.uniform(1e-1,1e3)
            k1is4 = np.random.uniform(1e-2,1e3)
            k2is4 = np.random.uniform(1e-2,1e3)
            kis4 = np.ndarray((numh,), buffer=np.array([k1is4,k2is4]), dtype = float)
            s4res = minimize(lmlh,kis4,args=(y,R),method = 'L-BFGS-B',jac=lmlgh,bounds = ((1e-2,1e3),(1e-2,1e3)),options={'maxiter':1e3})
            step4res = []
            if s4res.success:
                step4res.append(s4res.x)
            else:
                #raise ValueError(s4res.message)
                #k1is4 = np.random.uniform(1e-2,1e3)
                #k2is4 = np.random.uniform(2e-1,1e3)
                k1is4 = np.random.uniform(1e-2,1e3)
                k2is4 = np.random.uniform(1e-2,1e3)
                print("error in hypopth() - reinitialising hyperparameters")
                continue 
            k1s4[i] = step4res[0][0]
            k2s4[i] = step4res[0][1]
        lmltest = [lmlh([k1s4[i],k2s4[i]],y,R) for i in range(numrestarts)]
        k1f = k1s4[np.argmin(lmltest)]
        k2f = k2s4[np.argmin(lmltest)]
            #lml(params,y,sig)
        return k1f, k2f
    def hetloopSK(fmst,varfmst,numiters,numrestarts):
        s = 200
        #k1is3, k2is3, k1is4,k2is4  =  np.random.uniform(1e-2,1e2,4)
        MSE = np.empty([numiters,1])
        NLPD = np.empty([numiters,1])
        fmstf = np.empty([numiters,n])
        varfmstf = np.empty([numiters,n])
        lmloptf = np.empty([numiters,1])
        rf = np.empty([numiters,n])
        i = 0
        while i < numiters:        
            breakwhile = False
            # Step 2: estimate empirical noise levels z 
            #k1is4,k2is4  = np.random.uniform(1e-2,1e2,2)
            #k1is3, k1is4  =  np.random.uniform(1e-2,1e2,2)
            #k2is3, k2is4  =  np.random.uniform(1e-1,1e2,2)
            k1is3  =  np.random.uniform(1e-3,1e3,1)
            k2is3  =  np.random.uniform(1e-3,1e3,1)
            z = np.empty([n,1])
            for j in range(n):
                #np.random.seed()
                normdraw = normal(fmst[j], varfmst[j]**0.5, s).reshape(s,1)
                z[j] = np.log((1/s)*0.5*sum((y[j] - normdraw)**2))
                if math.isnan(z[j]): # True for NaN values
                    breakwhile = True
                    break
            if breakwhile:
                print("Nan value in z -- restarting iter "+ str(i))
                
                continue
            #  Step 3: estimate GP2 on D' - (x,z)
            kernel2 = C(k1is3, (1e-3, 1e3)) * RBF(k2is3, (1e-3, 1e3)) 
            gpr2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer = numrestarts, normalize_y=False, alpha=np.var(z))
            
            gpr2.fit(x, z)
            ystar2, sigma2 = gpr2.predict(x, return_std=True )
            sigma2 = (sigma2**2 + 1)**0.5
        # Step 4: train heteroscedastic GP3 using predictive mean of G2 to predict log noise levels r
            r = exp(ystar2)
            R = r*np.identity(n)
            k1s4, k2s4 = hypopth(y,numrestarts,R)
            fmst4, varfmst4, lmlopt4 = GPRfith(x,k1s4,k2s4,R,R)
            # test for convergence 
            MSE[i] = (1/n)*sum(((y-fmst4)**2)/np.var(y))
            #NLPD[i] = sum([(1/n)*(-np.log(norm.pdf(x[j], fmst4[j], varfmst4[j]**0.5))) for j in range(n) ])
            nlpdarg = np.zeros([n,1])
            #nlpdtest = np.zeros([n,1])
            for k in range(n):
                nlpdarg[k] = -np.log10(norm.pdf(x[k], fmst4[k], varfmst4[k]**0.5))
                #nlpdtest[k] = norm.pdf(x[k], fmst4[k], varfmst4[k]**0.5)
            #print("mean NLPD log arg " + str(nlpdtest) )
                #test3[k] = -np.log(norm.pdf(x[k], fmst[k], varfmst[k]**0.5))
            NLPD[i] = sum(nlpdarg)*(1/n)
            print("MSE = " + str(MSE[i]))
            print("NLPD = " + str(NLPD[i]))
            print("finished iteration " + str(i+1))
            fmstf[i,:] = fmst4.reshape(n)
            varfmstf[i,:] = varfmst4.reshape(n)
            lmloptf[i] = lmlopt4
            fmst = fmst4
            varfmst = varfmst4
            rf[i,:] = r.reshape(n)
            #k1is3 = k1s4
            #k2is3 = k2s4
            i = i + 1
        return fmstf,varfmstf, lmloptf, MSE, rf, NLPD #  , NLPD 
    #kernel1 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    #gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer = 0, normalize_y=True)
    kernel1 = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) 
    gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer = 10, normalize_y=False, alpha=np.var(y))
    gpr1.fit(x, y)
    ystar1, sigma1 = gpr1.predict(x, return_std=True )
    var1 = (sigma1**2 + np.var(y))
    #sigma1 = np.reshape(sigma1,(np.size(sigma1), 1))

    numiters = 20
    numrestarts = 5
    start_time = time.time()
    fmstf,varfmstf, lmlopt, mse, _,NLPD = hetloopSK(ystar1,var1,numiters,numrestarts)
    duration = time.time() - start_time

    ind = numiters - 1
    #ind = 
    fmst4 = fmstf[ind]
    varfmst4 = varfmstf[ind]
    
    sigs4 = varfmst4**0.5
    fmstps4 = fmst4 + sigs4
    fmst4i = scaler.inverse_transform(fmst4)
    fmstps4i = scaler.inverse_transform(fmstps4)
    
    #  ================================ Mutual information transform ===========================================
# =============================================================================
#     MIcalc = False # select whether to calculate MI using Guassian-Hermite quadrature 
#     # import constellation shapes from MATLAB-generated csv files 
#     if MIcalc:  
#         Qam4r = np.genfromtxt(open("qam4r.csv", "r"), delimiter=",", dtype =float)
#         Qam4i = np.genfromtxt(open("qam4i.csv", "r"), delimiter=",", dtype =float)
#         Qam16r = np.genfromtxt(open("qam16r.csv", "r"), delimiter=",", dtype =float)
#         Qam16i = np.genfromtxt(open("qam16i.csv", "r"), delimiter=",", dtype =float)
#         Qam32r = np.genfromtxt(open("qam32r.csv", "r"), delimiter=",", dtype =float)
#         Qam32i = np.genfromtxt(open("qam32i.csv", "r"), delimiter=",", dtype =float)
#         Qam64r = np.genfromtxt(open("qam64r.csv", "r"), delimiter=",", dtype =float)
#         Qam64i = np.genfromtxt(open("qam64i.csv", "r"), delimiter=",", dtype =float)
#         Qam128r = np.genfromtxt(open("qam128r.csv", "r"), delimiter=",", dtype =float)
#         Qam128i = np.genfromtxt(open("qam128i.csv", "r"), delimiter=",", dtype =float)
#         
#         Qam4 = Qam4r + 1j*Qam4i
#         Qam16 = Qam16r + 1j*Qam16i
#         Qam32 = Qam32r + 1j*Qam32i
#         Qam64 = Qam64r + 1j*Qam64i
#         Qam128 = Qam128r + 1j*Qam128i
#         #  ================================ Estimate MI ================================ 
#         # set modulation format order and number of terms used in Gauss-Hermite quadrature
# =============================================================================

#         M = 16
#         L = 6
#         
#         def MIGHquad(SNR):
#             if M == 4:
#                 Ps = np.mean(np.abs(Qam4**2))
#                 X = Qam4
#             elif M == 16:
#                 Ps = np.mean(np.abs(Qam16**2))
#                 X = Qam16
#             elif M == 32:
#                 Ps = np.mean(np.abs(Qam32**2))
#                 X = Qam32
#             elif M == 64:
#                 Ps = np.mean(np.abs(Qam64**2))
#                 X = Qam64
#             elif M == 128:
#                 Ps = np.mean(np.abs(Qam128**2))
#                 X = Qam128
#             else:
#                 print("unrecogised M")
#             sigeff2 = Ps/(10**(SNR/10))
#             Wgh = GHquad(L)[0]
#             Rgh = GHquad(L)[1]
#             sum_out = 0
#             for ii in range(M):
#                 sum_in = 0
#                 for l1 in range(L):      
#                     sum_inn = 0
#                     for l2 in range(L):
#                         sum_exp = 0
#                         for jj in range(M):  
#                             arg_exp = np.linalg.norm(X[ii]-X[jj])**2 + 2*(sigeff2**0.5)*np.real( (Rgh[l1]+1j*Rgh[l2])*(X[ii]-X[jj]));
#                             sum_exp = np.exp(-arg_exp/sigeff2) + sum_exp
#                         sum_inn = Wgh[l2]*np.log2(sum_exp) + sum_inn
#                     sum_in = Wgh[l1]*sum_inn + sum_in
#                 sum_out = sum_in + sum_out
#             return np.log2(M)- (1/(M*np.pi))*sum_out 
#         
#         def findMI(SNR):
#             with multiprocessing.Pool() as pool:
#                 Ixy = pool.map(MIGHquad, SNR) 
#             return Ixy
# =============================================================================
        
    print("HGP fitting duration: " + str(duration)) 
    
    return fmst4i, fmstps4i, lmlopt, mse, NLPD

numiters = 20

prmn = np.empty([numedges,numpoints])
prmnp = np.empty([numedges,numpoints])
lml = np.empty([numedges,numiters])
MSE = np.empty([numedges,numiters])
NLPD = np.empty([numedges,numiters])

for i in range(np.size(snr,0)):
    prmn[i], prmnp[i], lmls, MSEs, NLPDs = HGPfunc(x,snr[i],False)
    lml[i] = lmls.reshape(numiters)
    MSE[i] = MSEs.reshape(numiters)
    NLPD[i] = NLPDs.reshape(numiters)
#prmn, prmnp, lml, MSE, NLPD = HGPfunc(x,snr,False)
sig = (prmnp - prmn)

prmnp1 = prmn + sig    
prmnn1 = prmn - sig
prmnp2 = prmn + 2*sig
prmnn2 = prmn - 2*sig
prmnp3 = prmn + 3*sig    
prmnn3 = prmn - 3*sig
prmnp4 = prmn + 4*sig    
prmnn4 = prmn - 4*sig
prmnp5 = prmn + 5*sig    
prmnn5 = prmn - 5*sig


# %%
algtest = False
if algtest:

    font = { 'family' : 'sans-serif',
                    'weight' : 'normal',
                    'size'   : 15}
    matplotlib.rc('font', **font)

    f, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(x,snr,'+')
    ax.plot(x,prmn,color='k')
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([prmnp3,
                            (prmnn3)[::-1]]),
            alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
    # =============================================================================
    # ax.fill(np.concatenate([x, x[::-1]]),
    #          np.concatenate([prmnp3[ind],
    #                         (prmnp1[ind])[::-1]]),
    #          alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
    # ax.fill(np.concatenate([x, x[::-1]]),
    #          np.concatenate([prmnn1[ind],
    #                         (prmnn3[ind])[::-1]]),
    #          alpha=0.3, fc='r', ec='None')
    # =============================================================================
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([prmnp5,
                            (prmnp3)[::-1]]),
            alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([prmnn3,
                            (prmnn5)[::-1]]),
            alpha=0.3, fc='g', ec='None')

    ax2.plot(x, sig, '--', label="learned $\sigma$")
    ax2.plot(x, sd, '-', label="true $\sigma$")
    ax2.set_ylabel("$\sigma$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([x[0], x[-1]])
    #ax.set_ylim([6.4, 9.5])
    #ax.set_xticklabels(xlab)
    #ax.set_yticklabels(ylab)
    plt.legend(ncol=1)
    #plt.savefig('HGPtestgoldberg.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('HGPtestwilliams.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhaba.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(lml)
    plt.ylabel("LML")
    #plt.savefig('HGPtestgoldberglml.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('HGPtestwilliamslml.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhabalml.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    plt.plot(MSE)
    plt.ylabel("MSE")
    #plt.savefig('HGPtestgoldbergmse.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('HGPtestwilliamsmse.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhabamse.pdf', dpi=200,bbox_inches='tight')

    plt.show()

    plt.plot(NLPD)
    plt.ylabel("NLPD")
    #plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
    #plt.savefig('HGPtestyuanwhabanlpd.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    """ f, ax = plt.subplots()
    ax.plot(x, sig, '--', label="learned $\sigma$")
    ax.plot(x, sd, '-', label="true $\sigma$")
    ax.set_xlabel("x")
    ax.set_ylabel("$\sigma$")
    plt.savefig('HGPtestgoldbergsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() """


# %%

""" np.savetxt('hetprmnextdegst.csv', prmn, delimiter=',')
np.savetxt('hetsigextdegst.csv', sig, delimiter=',') """

font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
matplotlib.rc('font', **font)

ind = 2
f, ax = plt.subplots()
ax.plot(x,snr[ind],'+')
ax.plot(x,prmn[ind],color='k')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp3[ind],
                        (prmnn3[ind])[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# =============================================================================
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnp3[ind],
#                         (prmnp1[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnn1[ind],
#                         (prmnn3[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None')
# =============================================================================
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp5[ind],
                        (prmnp3[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
ax.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnn3[ind],
                        (prmnn5[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([x[0], x[-1]])
#ax.set_ylim([6.4, 9.5])
#ax.set_xticklabels(xlab)
#ax.set_yticklabels(ylab)
#plt.axis([-1,100,1.0,8.0])
plt.legend()
#plt.savefig('JOCNhetGP.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPextdegst.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPextdegstmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPextdegdvmoredata.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%
plt.plot(lml[ind])
plt.ylabel("LML")
#plt.savefig('HGPtestgoldberglml.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestwilliamslml.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestyuanwhabalml.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.plot(MSE[ind])
plt.ylabel("MSE")
#plt.savefig('HGPtestgoldbergmse.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestwilliamsmse.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestyuanwhabamse.pdf', dpi=200,bbox_inches='tight')

plt.show()

plt.plot(NLPD[ind])
plt.ylabel("NLPD")
#plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestgoldbergnlpd.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('HGPtestyuanwhabanlpd.pdf', dpi=200,bbox_inches='tight')
plt.show()



# 
# %%

# [160,480,960,1200, 1600, 2400]

f, ax = plt.subplots()
#xl = ['0','2','4', '6', '8', '10']
#ax.plot(x,sig[0],color='r',LineStyle=':',label='160 km')
ax.plot(x,sig[1],color='m',LineStyle='-.',label='480 km')
ax.plot(x,sig[2],color='g',LineStyle='-.',label= '960 km')
ax.plot(x,sig[3],color='b',LineStyle='-.',label= '1200 km')
ax.plot(x,sig[4],color='r',LineStyle='-.',label= '1600 km')
#ax.plot(x,sig[5],color='c',LineStyle='-.',label= '2400 km')

""" ax.plot(x,sig[0],color='r',LineStyle='-.',label='A')
ax.plot(x,sig[1],color='b',LineStyle='-.',label='B')
ax.plot(x,sig[2],color='g',LineStyle='-.',label= 'C') """


# for standard heteroscedastic experiment 
#sdhet = np.linspace(0.04,0.06,numpoints)
numyrsh = 200
# for simulation of TRx fault 
sdh1 = np.linspace(0.04,0.042,int(numyrsh/2)+1)
sdh2 = np.linspace(0.08,0.082,int(numyrsh/2)-1)
sdhet = np.append(sdh1,sdh2)

# for comparison of different variances 
""" sdhet1 = np.linspace(0.04,0.06,numpoints)
sdhet2 = np.linspace(0.04,0.08,numpoints)
sdhet3 = np.linspace(0.04,0.16,numpoints) """

ax.plot(x,sdhet,color='k',label='Eq. (1) $\sigma(t)$')
""" ax.plot(x,sdhet1,color='r',label='Eq. (1) $\sigma(t)$ A')
ax.plot(x,sdhet2,color='b',label='Eq. (1) $\sigma(t)$ B')
ax.plot(x,sdhet3,color='g',label='Eq. (1) $\sigma(t)$ C') """

#plt.plot(x,sigrf[ind],color='k',LineStyle='-',label='$\sqrt{r(x)}$ 1')

ax.set_xlabel("time (years)")
ax.set_ylabel("SNR $\sigma$ (dB)")
#ax2.set_ylabel("Shannon throughput gain (%)")
ax.set_xlim([x[0], x[-1]])
#ax.set_ylim([0.03, 0.12])
#ax.set_xticklabels(xl)
ax.legend(loc=2,ncol=2, prop={'size': 11})

#plt.savefig('JOCNhetgpsig.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsig20moredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsigdvmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsigst.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetgpsigstmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.axis([x[0],x[-1],0.03,0.09])
#plt.xticklabels()
plt.show()

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln2 = ax1.plot(x,snr[ind],'+')
ln3 = ax1.plot(x,prmn[ind],color='k')

ln4 = ax1.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp3[ind],
                        (prmnn3[ind])[::-1]]),
         alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# =============================================================================
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnp3[ind],
#                         (prmnp1[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
# ax.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([prmnn1[ind],
#                         (prmnn3[ind])[::-1]]),
#          alpha=0.3, fc='r', ec='None')
# =============================================================================
ln5 = ax1.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnp5[ind],
                        (prmnp3[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
ln6 = ax1.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([prmnn3[ind],
                        (prmnn5[ind])[::-1]]),
         alpha=0.3, fc='g', ec='None')
ln7 = ax2.plot(x,sig[ind],color='r',LineStyle='-.',label='$R(time)$')

ln8 = ax2.plot(x,sdhet,'--',color='k',label='Eq. (1) $\sigma(t)$')
    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("SNR (dB)")
ax2.set_ylabel("SNR $\sigma$ (dB)")
    
ax1.set_xlim([x[0], x[-1]])
#ax1.set_ylim([13.2,14.5])
#ax2.set_ylim([0.03,0.09])
#ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
lns = ln4+ln5+ln7+ln8
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs,loc=4, ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('hetGPexampledvmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPexamplestmoredata.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPexamplest.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('hetGPexample20moredata.pdf', dpi=200,bbox_inches='tight')

plt.show()

# %%

plt.plot(x,snr[1],'+')

plt.show()


# %%
