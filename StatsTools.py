import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA


def LSUni(X1,X2):
    """Returns the linear least squares coeffs and data fit"""
    
    # Obtain the primary mean
    mu1 = np.mean(X1)
    mu2 = np.mean(X2)
    
    # Now compute the regression coefficents
    temp1 = X1-mu1
    temp2 = X2-mu2
    
    b1 = np.sum(temp1*temp2)/(np.sum(temp1*temp1))
    b0 = mu2 - (b1*mu1) 
    
    # Now form a new array that represents the fit
    X3 = b0+(b1*X1)
    
    return b0, b1, X3

def LSMulti(X, Y):
    """Return the multiple least squares coeffs and data fit"""
    return (np.matmul(np.linal.inv(np.matmul(np.transpose(X),X)), np.mmatmul(np.matrix.transpose(X), Y)))


def covar(X,Y):
    """Return the sample covariance of the passed in series, X"""
    # Define the lenght of the time series
    N = X.shape[0]

    # Get the mean of the series
    mu_X = np.mean(X)
    mu_Y = np.mean(Y)
    # Now make a translated and product series
    diffs_X = X-mu_X
    diffs_Y = Y-mu_Y
    # Now make a produce series
    prod_XY = diffs_X*diffs_Y
    return np.sum(prod_XY)*(1/(N-1)) 

def corr(X,Y):
    """Return the correlation coefficent of the two passed series X and Y"""
    return covar(X,Y)*(1/(np.std(X)*np.std(Y)))

def regvar(X):
    """Return the sample variance of the passed in series, X"""
    return np.sum((X-np.mean(X))*(X-np.mean(X)))/(X.shape[0]-1)

def atcovar(X, max_k=False):
    """Return the autocovariance of the passed in series, X"""
    if max_k:
        K = np.arange(0,max_k+1)
    else:
        K = np.arange(0,X.shape[0],1)
    ret_arr = np.zeros((len(K)))
    for i in range(0, len(K)):
        for j in range(0, X.shape[0]-K[i]):
            ret_arr[i] += (X[j]-np.mean(X))*(X[j+K[i]]-np.mean(X))
        ret_arr[i] = ret_arr[i]*(1/X.shape[0])
    return ret_arr

def atcorr(X,max_k=False):
    """Return the autocorrelation function of the passed in series, X"""
    # First get the autocovariance 
    if max_k:
        acv = atcovar(X, max_k)
    else:
        acv = atcovar(X)
    # The autocorrelation is simply normalized by the initial term in the acv
    return acv*(1/acv[0])


def plotACF(X, max_k=False):
    """Plot the autocorrelation function"""
    # First we need the autocorrelation function if we are going to plot it
    if max_k:
        acf = atcorr(X,max_k)
    else:
        acf = atcorr(X)

    # Now instantiate a figure to plot it on
    PLOT = plt.figure(figsize=(10,5), dpi=85)
    AX = PLOT.add_subplot(1,1,1)

    if max_k:
        AX.stem(np.arange(0,max_k+1,1), acf[0:max_k+1])
    else:
        AX.stem(np.arange(0,X.shape[0]+1,1), acf)


def AutoRegFit(X, order):
    """Use statsmodels package to estimate an autoregressive model to the inputted data source"""
    model = AutoReg(X, order).fit()

    return model, model.summary()


def MovAveFit(X, order):
    """Use statsmodels package to etimate the moving average model to the inputted data source"""
    model = ARIMA(X, order=(0,0,order)).fit()
    
    return model, model.summary()


def ARMAFit(X, p, q):
    """Use ststamodels package to estimate the ARMA model to the inputted dat source"""
    model = ARIMA(X, order=(p, 0, q)).fit()
    
    return model, model.summary()


def ARIMAFit(X, p, d, q):
    model = ARIMA(X, order=(p,d,q)).fit()
    
    return model, model.summary()


def resAcf(sample, model):
    """Display the ACF of the residuals to test for iid property"""
    residuals = sample-model
    # Now get the ACF of the residuals 
    plotAcf(residuals, 20)
    
    return residuals


def ARIMAAIC(X, max_p,max_d,max_q):
    """Run an AIC measure analysis on the model fit on the possible (p,d,q) triplets"""
    
    P = np.arange(1,max_p+1,1)
    D = np.arange(0,max_d+1,1)
    Q = np.arange(1,max_q+1,1)
    
    min_AIC = None
    min_triple = None
    
    for p in P:
        for d in D:
            for q in Q:
                print(f"({p},{d},{q})")
                model, model_summary = ARIMAFit(X, p, d, q)
                AIC = model.aic
                
                if min_AIC is None:
                    min_AIC = AIC
                    min_triple = np.array([p,d,q])
                else:
                    if AIC < min_AIC:
                        min_AIC = AIC
                        min_triple = np.array([p,d,q])
    
    return min_triple, min_AIC
    
        
    

def ARAIC(X, max_p, plot_out):
    """Run an AIC measure analysis on the model fit up to order-p"""

    # Make an array containg the possible model orders up to the level of p
    O = np.arange(0, max_p+1)

    AICs = np.zeros(len(O))

    for i in range(0, len(O)):
        # Generate the model using the designated fitting function
        model_order = O[i]
        ret_model, ret_model_summary = AutoRegFit(X, model_order)
        # Get the AIC of the current model
        ret_AIC = ret_model.aic
        AICs[i] = ret_AIC
    
    # Check if we need to plot the AICs before return the array
    if plot_out:
        fig = plt.figure(figsize=(10,10), dpi=85)
        ax = fig.add_subplot(1,1,1)
        ax.plot(O, AICs)
        ax.set_title('AIC Function', fontdict={'size': 16})
        ax.set_xlabel('Autoregressive Model Order', fontdict={'size': 14})
        ax.set_ylabel('AIC Measure', fontdict={'size': 14})    

    return np.vstack((O, AICs))


        
'''
Z = np.random.randn(100)

X = np.zeros(100)
X[0] = Z[0]
X[1] = 0.666*X[0] + Z[1]
alpha_1 = 0.666
alpha_2 = -0.333
for i in range(2,len(X)):
    X[i] = ((alpha_1*X[i-1])+(alpha_2*X[i-2]))+Z[i]

#plt.plot(np.arange(1,len(X)+1,1), X)
#plotAcf(X, 20)
mod, mod_sum = AutoRegFit(X, 2)
print(mod.summary())

#results = AR_AIC(X, 25, True)
#print(results)

sims1 = simAR(mod, Z, 100, 1)
'''

# Read in the data 
data = np.genfromtxt("C:/Users/Joshu/Downloads/shampoo.csv", delimiter=',')

#min_triple, min_AIC = ARIMAAIC(lrt, 4, 1, 4)

#model, model_summary = ARIMAFit(lrt, 1, 1, 2)

#print(model_summary)
