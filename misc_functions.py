import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
from scipy.linalg import sqrtm
from scipy.optimize import fsolve
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from scipy.stats import t
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Example PDF for X (standard normal distribution)



def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    return alpha, beta
# Preparation of the routine
def expand_vector(vec,length):
    return np.repeat(vec[:,np.newaxis,:], length, axis=1)


def GA_IRB_MC(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          q = 0.999, # quantile level
          N_sim = 10000000,
          constant_rho = False,
          constant_rho_val = 0.35,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10
    b = (0.11852-0.05478*np.log(np.maximum(PD,eps)))**2 # Maturity adjustment
    MA = (1/(1-1.5*b))*(1+(M-2.5)*b)
    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta   
    EAD = EAD/np.sum(EAD)
    X = np.random.normal(size = (N_sim,1))
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
    epsilon = np.random.normal(size = (N_sim,len(PD)))
    threshold =  norm.ppf(PD)
    threshold = np.reshape(threshold,(1,len(threshold)))
    if constant_rho:
        rho= np.array([constant_rho_val]*len(PD))
    else:
        rho = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))
    rho = np.reshape(rho,(1,len(rho)))
    Y = np.sqrt(rho) * X + np.sqrt(1 - rho) * epsilon
    D =  Y<threshold
    loss = np.sum(EAD*LGD*D*MA,1)
    
    VaR = np.quantile(loss,q)
    cond_loss = np.sum(ELGD*EAD*MA*norm.cdf(np.sqrt(1/(1-rho))*(norm.ppf(PD)+np.sqrt(rho)*norm.ppf(q))))
    GA = VaR - cond_loss

    return GA

def GA_IRB_MC2(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          q = 0.999, # quantile level
          N_sim = 10000000,
          constant_rho = False,
          constant_rho_val = 0.35,
          rho_input = False,
          rho_input_val = 0,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10
    b = (0.11852-0.05478*np.log(np.maximum(PD,eps)))**2 # Maturity adjustment
    MA = (1/(1-1.5*b))*(1+(M-2.5)*b)
    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta   
    EAD = EAD/np.sum(EAD)
    X = np.random.normal(size = (N_sim,1))
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
    epsilon = np.random.normal(size = (N_sim,len(PD)))
    threshold =  norm.ppf(PD)
    threshold = np.reshape(threshold,(1,len(threshold)))
    if constant_rho:
        rho= np.array([constant_rho_val]*len(PD))
    else:
        rho = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))
    if rho_input:
        rho = rho_input_val
    rho = np.reshape(rho,(1,len(EAD)))
    Y = np.sqrt(rho) * X + np.sqrt(1 - rho) * epsilon
    D =  Y<threshold
    loss = np.sum(EAD*LGD*D*MA,1)
    
    VaR = np.quantile(loss,q)
    cond_loss = np.sum(ELGD*EAD*MA*norm.cdf(np.sqrt(1/(1-rho))*(norm.ppf(PD)+np.sqrt(rho)*norm.ppf(q))))
    GA = VaR - cond_loss

    return GA,cond_loss


def GA_IRB_MC_t_distribution(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          q = 0.999, # quantile level
          N_sim = 10000000,
          df = 4,
          constant_rho = False,
          constant_rho_val = 0.35,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10
    b = (0.11852-0.05478*np.log(np.maximum(PD,eps)))**2 # Maturity adjustment
    MA = (1/(1-1.5*b))*(1+(M-2.5)*b)
    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta   
    EAD = EAD/np.sum(EAD)
    
    
    X = t.rvs(df =df, size = (N_sim,1))
    
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
   
    def f_X(x):
        return t.pdf(x,df=df)

    def F_Yn(y, rho_n):
        integrand = lambda x: norm.cdf((y - np.sqrt(rho_n) * x) / np.sqrt(1 - rho_n)) * f_X(x)
        result, _ = quad(integrand, -np.inf, np.inf)
        return result

    def inverse_F_Yn(target, rho_n, y_bounds=(-20, 1)):
        """
        Numerically computes the inverse of F_Yn for a given target value.

        Parameters:
        target: The value for which to find the inverse.
        rho_n: The parameter \rho_n in the function.
        y_bounds: Tuple indicating the search interval for y.

        Returns:
        y such that F_Yn(y) = target
        """
        def objective(y):
            return F_Yn(y, rho_n) - target
        
        try:
            solution = root_scalar(objective, bracket=y_bounds, method='bisect')
            return solution.root
        except ValueError:
            return -20



    
    if constant_rho:
        rho= np.array([constant_rho_val]*len(PD))
    else:
        rho = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))
        
    
    inverse_value = np.array([inverse_F_Yn(PD[i], rho[i]) for i in range(len(rho))])
    rho = np.reshape(rho,(1,len(rho)))
    
    epsilon = norm.rvs(size = (N_sim,len(PD)))
    threshold =  inverse_value
    threshold = np.reshape(threshold,(1,len(PD)))
    
    Y = np.sqrt(rho*(df-2)/df) * X + np.sqrt(1-rho) * epsilon
    D =  Y<threshold
    loss = np.sum(EAD*LGD*D*MA,1)
    
    
    
    VaR = np.quantile(loss,q)
    
    cond_pd = norm.cdf(1/np.sqrt(1-rho)* (inverse_value-np.sqrt(rho)*t.ppf(1-q,df=df)) )
    #cond_pd = cauchy.cdf((1/(1-np.sqrt(rho))*(threshold+np.sqrt(rho)*cauchy.ppf(q,loc=0, scale=1/2))),loc=0, scale=1/2)
    cond_loss = np.sum(ELGD*EAD*MA*cond_pd)
    GA = VaR - cond_loss

    return GA, cond_loss


def GA_IRB_MC_Cauchy(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          q = 0.999, # quantile level
          N_sim = 10000000,
          constant_rho = False,
          constant_rho_val = 0.35,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10
    b = (0.11852-0.05478*np.log(np.maximum(PD,eps)))**2 # Maturity adjustment
    MA = (1/(1-1.5*b))*(1+(M-2.5)*b)
    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta   
    EAD = EAD/np.sum(EAD)
    
    
    X = cauchy.rvs(loc=0, scale=1/2, size = (N_sim,1))
    
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
        
    epsilon = norm.rvs(size = (N_sim,len(PD)))
    #epsilon = np.random.normal(size = (N_sim,len(PD)))
    threshold =  cauchy.ppf(PD,loc=0, scale=1/2)
    threshold = np.reshape(threshold,(1,len(threshold)))
    if constant_rho:
        rho= np.array([constant_rho_val]*len(PD))
    else:
        rho = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))
    rho = np.reshape(rho,(1,len(rho)))
    Y = cauchy.ppf(norm.cdf(np.sqrt(rho)*norm.ppf(cauchy.cdf(X,loc=0, scale=1/2))+(np.sqrt(1-rho))*epsilon),loc=0, scale=1/2)
    #    Y = np.sqrt(rho) * X + (1-np.sqrt(rho)) * epsilon
    D =  Y<threshold
    loss = np.sum(EAD*LGD*D*MA,1)
    
    VaR = np.quantile(loss,q)
    
    cond_pd = norm.cdf(1/np.sqrt(1-rho)* (norm.ppf(PD)+np.sqrt(rho)*norm.ppf(q)) )
    #cond_pd = cauchy.cdf((1/(1-np.sqrt(rho))*(threshold+np.sqrt(rho)*cauchy.ppf(q,loc=0, scale=1/2))),loc=0, scale=1/2)
    cond_loss = np.sum(ELGD*EAD*MA*cond_pd)
    GA = VaR - cond_loss

    return GA, cond_loss



def GA_IRB_MC_twofactor_Cauchy(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          RHO,  # Vector of Correlations
          q = 0.999, # quantile level
          N_sim = 10000000,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10    
    EAD = EAD/np.sum(EAD)
    X = cauchy.rvs(loc=0, scale=1/2, size = (N_sim,2,1))
    MA = 1
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
    epsilon = norm.rvs(size = (N_sim,len(PD)))
    threshold =  cauchy.ppf(PD,loc=0, scale=1/2)
    threshold = np.reshape(threshold,(1,len(threshold)))
    RHO = np.reshape(RHO,(1,2,len(PD)))
    Y = cauchy.ppf(norm.cdf(np.sum(RHO*norm.ppf(cauchy.cdf(X,loc=0, scale=1/2)),axis =1)+np.sqrt(1-np.sum(RHO**2,axis=1))*epsilon),loc=0, scale=1/2)
    #Y = np.sum(RHO* X,axis = 1) + (1 - np.sum(RHO,axis=1)) * epsilon 
    D =  Y<threshold
    loss = np.sum(MA*EAD*LGD*D,1)
    
    VaR = np.quantile(loss,q)
    
    cond_PD = norm.cdf(np.sqrt(1/(1-np.sum(RHO**2,axis=1)))*(norm.ppf(PD)+np.sum(RHO* norm.ppf(cauchy.cdf(X,loc=0, scale=1/2)),axis = 1)))
    cond_loss_sum = np.sum(ELGD*EAD*MA*cond_PD,axis = 1)
    #cond_loss_sum = np.sum(ELGD*EAD*MA*cauchy.cdf(np.sqrt(1/(1-np.sum(RHO,axis=1)))*(threshold+np.sum(RHO* X,axis = 1)),loc=0, scale=1/2),axis = 1)
    cond_loss = np.quantile(cond_loss_sum,q)
    GA = VaR - cond_loss

    return GA, cond_loss


def GA_IRB_MC_twofactor(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          RHO,  # Vector of Correlations
          q = 0.999, # quantile level
          N_sim = 10000000,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10    
    EAD = EAD/np.sum(EAD)
    X = np.random.normal(size = (N_sim,2,1))
    MA = 1
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
    epsilon = np.random.normal(size = (N_sim,len(PD)))
    threshold =  norm.ppf(PD)
    threshold = np.reshape(threshold,(1,len(threshold)))
    RHO = np.reshape(RHO,(1,2,len(PD)))
    Y = np.sum(RHO* X,axis = 1) + np.sqrt(1 - np.sum(RHO**2,axis=1)) * epsilon 
    D =  Y<threshold
    loss = np.sum(MA*EAD*LGD*D,1)
    
    VaR = np.quantile(loss,q)
    
    cond_loss_sum = np.sum(ELGD*EAD*MA*norm.cdf(np.sqrt(1/(1-np.sum(RHO**2,axis=1)))*(norm.ppf(PD)+np.sum(RHO* X,axis = 1))),axis = 1)
    cond_loss = np.quantile(cond_loss_sum,q)
    GA = VaR - cond_loss

    return GA, cond_loss

def GA_comparable_one_factor(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          EAD, # Vector of Exposures at default,
          RHO,  # Vector of Correlations
          q = 0.999, # quantile level
          N_sim = 10000000,
          nu = 0.25,
          LGD_constant = False,
          M = 1):
    eps = 1e-10    
    EAD = EAD/np.sum(EAD)
    

    
    

    MA = 1
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD,nu*(ELGD)*(1-ELGD))
        LGD = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (len(EAD)))for _ in range(N_sim)])) #(Nsim, 1, len(EAD))
    else:
        LGD = ELGD
        

    
    # Determination of b       
    r = np.reshape(np.sum(RHO,axis =1)**2,len(PD))
    
    alpha = np.reshape(RHO/np.reshape(np.repeat(r, 2),(1,2,len(PD))),(2,len(PD)))
    c = LGD*EAD*norm.cdf((norm.ppf(PD)+r*norm.ppf(q))/(np.sqrt(1-r**2)))
    b = np.sum(c*alpha,axis=1)
    lam = np.sum(b**2)
    b = np.reshape(b/np.sqrt(lam),(2,1))   
    rho = np.sum(np.reshape(RHO,(2,len(PD)))*b,axis = 0)   
    
    X = np.random.normal(size = (N_sim,1))
    epsilon = np.random.normal(size = (N_sim,len(PD)))
    threshold =  norm.ppf(PD)
    threshold = np.reshape(threshold,(1,len(threshold)))
    rho = np.reshape(rho,(1,len(PD)))**2
    
    Y = np.sqrt(rho) * X + np.sqrt(1 - rho) * epsilon
    D =  Y<threshold
    loss = np.sum(EAD*LGD*D*MA,1)
    
    VaR = np.quantile(loss,q)
    cond_loss = np.sum(ELGD*EAD*MA*norm.cdf(np.sqrt(1/(1-rho))*(norm.ppf(PD)+np.sqrt(rho)*norm.ppf(q))))
    GA = VaR - cond_loss
    return GA, cond_loss    

def GA_MC_actuarial(LGD_Vector,EAD_Vector,PD_Vector,rho_Vector,q,xi,nu,N_sim,scen_cond_exp,max_obligors = 100):
    
    X = np.random.gamma(shape = xi,scale = 1/xi, size = (N_sim))
    X = np.sort(X)
    quantile_loss_X = np.quantile(X,q)
    quantile_index_X= (np.searchsorted(X,quantile_loss_X))-1    
    ind_upper = int(np.min([quantile_index_X+scen_cond_exp/2,N_sim]))
    ind_lower = int(np.max([quantile_index_X-scen_cond_exp/2,0]))    
    X = np.repeat(X[np.newaxis,:], 1, axis=0 )# Dimension: (B,N_sim,max_obligors)
    X = np.repeat(X[:,:,np.newaxis], max_obligors, axis=2) # Dimension: (B,N_sim,max_obligors)
    
    #Create the Input
    Input = np.concatenate([LGD_Vector,EAD_Vector,PD_Vector,rho_Vector],axis =1)
    alpha_2, beta_2 = parameters_beta(LGD_Vector,nu*(LGD_Vector)*(1-LGD_Vector))
    LGD_Vector = np.nan_to_num(np.array([np.random.beta(alpha_2,beta_2,size = (1,max_obligors))for _ in range(N_sim)])).transpose(1,0,2) 
    #LGD_Vector = expand_vector(LGD_Vector,N_sim) # Dimension: (B,N_sim,max_obligors)
    EAD_Vector = expand_vector(EAD_Vector,N_sim) # Dimension: (B,N_sim,max_obligors)
    PD_Vector = expand_vector(PD_Vector,N_sim) # Dimension: (B,N_sim,max_obligors)
    rho_Vector = expand_vector(rho_Vector,N_sim)    # Dimension: (B,N_sim,max_obligors)
    #Compute the defaults
    pi_X = np.minimum(1,PD_Vector*(1+rho_Vector*(X-1)))
    D = np.random.binomial(1, pi_X)
    #Compute the loss
    loss = np.sum(EAD_Vector*LGD_Vector*D,2) # Dimension: (B,N_sim)
    quantile_loss_L = np.quantile(loss,q,axis = 1) # Dimension: (B)
    conditional_loss = np.mean(loss[:,(ind_lower):(ind_upper)],1)
    GA = (quantile_loss_L-conditional_loss)
    
    return GA


def GA_GL(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          A, # Vector of Exposures at default,
          M, # Vector of Maturities
          q = 0.999, # quantile level
          xi = 0.25, # precision parameter,
          nu = 0.25,# recovery parameter
          rho = 0.2, #omega parameter
          simplified = False,
          second_order = False,
          LGD_constant = False,
          IRB = False,
          constant_rho = False,
          constant_rho_val = 0.35):
    #Quantile
    alpha_x = gamma.ppf(q,a =xi, scale = 1/xi) # Quantile of a Gamma distribution with mean 1, variance 1/xi
    # Convert to numpy arrays:
    PD, ELGD, A, M = np.array(PD), np.array(ELGD), np.array(A), np.array(M)
    # Exposure shares
    s = A/np.sum(A) 
    # Variance of LGD
    VLGD = nu*ELGD*(1-ELGD)
    if LGD_constant:
        VLGD = 0
    # Parameter C
    C = (VLGD+ELGD**2)/ELGD
    # Loan loss reverse requirement
    R = ELGD*PD
    delta = (alpha_x-1)*(xi+(1-xi)/alpha_x) # Formula (15)
    # UL capital requirement (IRB Formula, TO CHECK AGAIN!)
    b = (0.11852-0.05478*np.log(np.maximum(PD,1e-10)))**2 # Maturity adjustment

    if IRB:
        R_corr = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50))) 
    else:
        def to_solve(R_corr):
            #R_corr = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))
            K = ((ELGD)*norm.cdf(np.sqrt(1/(1-R_corr))*norm.ppf(PD)+np.sqrt(R_corr/(1-R_corr))*norm.ppf(q))-PD*ELGD)*(1/(1-1.5*b))*(1+(M-2.5)*b)
            alpha_X =  gamma.ppf(q,a =xi, scale = 1/xi)
            return ELGD*PD*rho*(alpha_X-1)-K
        R_corr = fsolve(to_solve,[0.18]*len(A))
    if constant_rho:
        R_corr = constant_rho_val
    
    K = ((ELGD)*norm.cdf(np.sqrt(1/(1-R_corr))*norm.ppf(PD)+np.sqrt(R_corr/(1-R_corr))*norm.ppf(q))-PD*ELGD)*(1/(1-1.5*b))*(1+(M-2.5)*b)
    
    # Scale K for different quantiles than 99.9 %
    #K = K*(gamma.ppf(q,a =xi, scale = 1/xi)-1)/(gamma.ppf(0.999,a =xi, scale = 1/xi)-1)
    K_star = np.sum(K*s)
    GA2nd = 0
    #Simplified Approach
    if simplified:
        Q = (delta*(K+R)-K)
        GA1st = (1/(2*K_star))*np.sum((s**2)*C*Q)
    
    # Approach with all terms
    elif simplified == False:
        Term1 = delta*C*(K+R)+(delta*(K+R)**2)*(VLGD/(ELGD**2))
        Term2 = K*(C+2*(K+R)*(VLGD/(ELGD**2)))
        GA1st = (1/(2*K_star))*np.sum((s**2)*(Term1-Term2)) 
        
    if second_order == True:
        ELGD3 = ELGD*(ELGD*(1/nu-1)+1)*(ELGD*(1/nu-1)+2)/((1/nu)*(1/nu+1)) # 3rd moment of a BETA DISTRIBUTION?
        #################
        if LGD_constant:
            ELGD3 = ELGD**3 
        
        eta1 = ELGD3*((K+R)/(ELGD)+3*((K+R)**2)/(ELGD**2)+((K+R)**3)/(ELGD**3))
        eta2 = -3*(C*(K+R)+((K+R)**2)*VLGD/(ELGD**2))*(K+R)-(K+R)**3
        eta = eta1+eta2      

        #eta_prime1 = ELGD3*(-ELGD*K/((K+R)**2)+ 6*((K+R)*K)/(ELGD**2)-3*(((K+R)**2)*K)/(ELGD**3))
        eta_prime1 = ELGD3*(K/(ELGD)+ 6*((K+R)*K)/(ELGD**2)+3*(((K+R)**2)*K)/(ELGD**3))
        eta_prime2 =  -6*C*(K+R)*K-9*((K+R)**2)*K *VLGD/(ELGD**2) - 3*((K+R)**2) *K
        eta_prime = (1/(alpha_x-1))*(eta_prime1+eta_prime2)
        
        #eta_prime_prime1 = ELGD3*((2*(K**2)* ELGD)/((K+R)**3) +6*(K**2)/(ELGD**2)-6*((K+R)*(K**2))/(ELGD**3))
        eta_prime_prime1 = ELGD3*((6*(K**2)/(ELGD**2)+6*((K+R)*(K**2))/(ELGD**3)))
        eta_prime_prime2 = -6*(K**2)*(K+R)*VLGD/(ELGD**2)-6*(K**2)*(K+R)
        eta_prime_prime3 = -6*(C*(K**2)+2*(K**2)*(K+R)*VLGD/(ELGD**2))
        eta_prime_prime = (1/((alpha_x-1)**2))*(eta_prime_prime1+eta_prime_prime2+eta_prime_prime3)        
        
        Factor1 = ((alpha_x-1)**2)/(6*(K_star**2))
        h_prime_over_h = -(xi+(1-xi)/alpha_x)
        h_prime_prime_over_h = xi**2 - 2*(xi*(xi-1))/alpha_x + (xi-1)*(xi-2)/(alpha_x**2)
        Sum1 = np.sum((s**3)*(eta_prime_prime+2*h_prime_over_h*eta_prime+eta*h_prime_prime_over_h))
        GA2nd = Factor1*Sum1
    return GA1st + GA2nd

def GA_GM(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          A, # Vector of Exposures at default,
          M, # Vector of Maturities
          q, # quantile level          
         r, # interest rate
         T, # Maturity
         g, # current states
         trans_prob,#N times S matrix
         psi = 0.4, # Market Sharpe Ratio,
         rho =0.2,
         nu = 0.25,
         c = 0.05,
         S = 10, # Number of States of Rating
         second_order = False,
         default_only = False,
          LGD_constant = False): 
     
    
    
    N = len(A)  # Number of obligors
    threshold = 2**10# Instead of Inf for function C
    alpha_X = norm.ppf(1-q) # q-Quantile of X
    
    
    tenor = [np.array([i*0.5 for i in range((int(M[j]))*2+1)]) for j in range(N)]
    tenor = np.array(tenor,dtype = object)
    
    
    M = np.round(M*2,0)/2 # round to 0.5 digits
    
    #Convert to numpy-arrays
    PD, ELGD, A, M, g = np.array(PD), np.array(ELGD), np.array(A), np.array(M), np.array(g)
    
    #Create a matrix corresponding to g
    g_matrix = np.zeros((N,S+1))
    for i in range(N):
        g_matrix[i,g[i]] = 1
    
    # Normalize exposures, and consider weights
    A = A/np.sum(A)
    
    def convert_matrix_order(matrix_0):  # Function to rearrange the transition matrix s.t. s=0 corresponds to default
        reverse_indices = list(matrix_0.columns)[::-1]
        matrix_0 = matrix_0.reindex(columns=reverse_indices)
        matrix_0 = matrix_0.reindex(reverse_indices)
        return matrix_0
    
    if default_only == True: # No transitions allowed
        matrix_0 = convert_matrix_order(trans_prob)
        indices = matrix_0.columns
        sums = matrix_0.apply(lambda x: np.sum(x[1:]),1)
        converted_trans_prob = pd.DataFrame(np.diag(sums), index = indices, columns = indices)
        converted_trans_prob['D'] = matrix_0['D']
    else:        
        # Change the order of the columns and rows of the transition matrix
        converted_trans_prob = convert_matrix_order(trans_prob) #matrix where s=0: Default ~ s=1:B ~ s=2: A
    converted_trans_prob_half_yr = sqrtm(converted_trans_prob)
 
    
    
    def C(g,s): # Determination of bounds for default, g vector, s scalar
        eps = 1e-10         
        if s == S:
            return np.array([threshold]*N)
        if s == -1:
            return np.array([-threshold]*N)     
        else:
            val = np.sum(np.matmul(g_matrix,converted_trans_prob.iloc[:S+1,:s+1]),1)
            val = np.maximum(np.minimum(val,1-eps),eps)
            val = norm.ppf(val) 
            val = np.maximum(np.minimum(val, np.array([threshold]*N)),np.array([-threshold]*N))
        return val
        
        
    

    def p(s,t1,t2): #probability of default between t and T
        if isinstance(s, int): # if s is an integer number
            if t1 == t2:
                if s == 0:
                    return 1
                else:
                    return 0
            else:
                state = np.zeros(S+1)
                state[s] = 1 # indicate state s
                result = np.matmul(state,np.array(converted_trans_prob_half_yr))
                for i in range(max(int((t2-t1-0.5)*2),0)): 
                    result = np.matmul(result,np.array(converted_trans_prob_half_yr))
                return np.maximum(result[0],0)
                
        else: # if s is a vector
            if t1 == t2:
                return np.array([int(s[i]==0) for i in range(len(s))])
            else:
                state = np.zeros((len(s),S+1))
                for i in range(len(s)):
                    state[i,s[i]] = 1 # indicate state s
                result = np.matmul(state,np.array(converted_trans_prob_half_yr))
                for i in range(max(int((t2-t1-0.5)*2),0)):  
                    result = np.matmul(result,np.array(converted_trans_prob_half_yr))
                return np.maximum(result[:,0],0)    
            
            
    def ps_star(s,t1,t2):
        return norm.cdf(norm.ppf(p(s,t1,t2))+psi*np.sqrt(t2-t1)*np.sqrt(rho))  
        
    # def F(t):  #Forward value of the bond
    #     #Eva
    #     Term1 =c*0.5*np.sum(np.exp(-r*(t-tenor[:,int(t*2):])),1)
    #     Term2 = np.exp(-r*(tenor[:,-1]-t))
    #     return Term1 +Term2 
   
    def F_vector(t):  #Forward value of the bond
        #Eva
        Term1 =c*0.5*np.sum([[np.exp(-r(tenor[i][j]-t))*(tenor[i][j]>=t) for j in range(len(tenor[i]))] for i in range(N)],1)
        #Term1 =  c*0.5*np.array([np.sum([np.exp(-r(t_i-t)) for t_i in tenor[j][1:] if t_i >= t]) for j in range(N)])
        Term2 = np.array([np.exp(-r(tenor[j][-1]-t)) for j in range(N)])
        return Term1 + Term2
    
    # Compute once the F values
    #tenor_list = [i*0.5 for i in range((int(np.max(M)))*2+1)]
    #F_vec = [F_vector(t) for t in tenor_list]
    
    def F(t):
        return F_vector(t)
        #return F_vec[int(np.minimum(2*t,np.max(tenor_list)))]
    
        
    def P_0():
        Term1 = F(0)
        # Eva:
        Term2 = -np.array([np.sum([np.exp(-r(tenor[j][i]))*(ps_star(g,0,tenor[j][i])[j] \
                        -ps_star(g,0,tenor[j][i-1])[j])*(F(tenor[j][i])[j]*ELGD[j]) for i in range(1,len(tenor[j]))]) for j in range(N)])
        # Gordy:
        #Term2 = -np.sum([np.exp(-r*(tenor[i]))*(ps_star(g,0,tenor[i]) \
        #                -ps_star(g,0,tenor[i-1]))*(F(tenor[i])-(1-ELGD)*(1+c/2)) for i in range(1,len(tenor))],0)
        val = Term1 + Term2
        return val #Bond price for each obligor (N-dim vector)
    
    # Compute the value P_0 once.
    P_00 = P_0() 
    
    def lambda_vector(s):
        if s >0:
            #Eva:
            Term1 = c*0.5*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T]) for j in range(N)])
            Term2 = F(T)
            Term3 = - np.array([np.sum([np.exp(-r(tenor[j][i]-T))*(ps_star(s,0,tenor[j][i]-T)[j] \
                            -ps_star(s,0,tenor[j][i-1]-T)[j])*(ELGD[j])*F(tenor[j][i])[j] for i in range(1,len(tenor[j])) if tenor[j][i]>T]) for j in range(N)])
                #Gordy
            #Term1 = c*0.5*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <=T])
            #Term2 = F(T)
            #Term3 = np.sum([np.exp(-r*(tenor[i]-T))*(ps_star(s,0,tenor[i]-T) \
            #            -ps_star(s,0,tenor[i-1]-T))*(F(tenor[i])-(1-ELGD)*(1+c/2)) for i in range(1,len(tenor)) if tenor[i]>T],0)
    
            lam = (1/ P_00)*(Term1 + Term2 +Term3)           
    
        elif s==0:
            if LGD_constant and default_only:
                lam = (1-ELGD)
            else:
                #Eva:
                lam = (1/ P_00)*((c/2)*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T])for j in range(N)])+(1-ELGD)*F(T)) 
            #Gordy:
            #lam = (1/ P_0())*((c/2)*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <T])+(1-ELGD)*(1+c/2))
        return lam
    
    lambda_vec = [lambda_vector(s) for s in range(S+1)]
                  
    
    def lambda_(s):
        return lambda_vec[s]
    
    

    def VLGD_squared():
        if LGD_constant:
            return 0
        else:
            return nu*(np.mean(ELGD))*(1-np.mean(ELGD))
    
    def xi_0squared():
        # Eva
        return  (F(T)/(P_00))**2 *VLGD_squared()
        #Gordy
        #return  ((1+c/2)/P_0())**2 *VLGD_squared()

    def pi_(x,s):
        Term1 = norm.cdf((C(g,s)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        Term2 =  -norm.cdf((C(g,s-1)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        return Term1 + Term2
    
    
    def pi_prime(x,s):
        Term1 = -np.sqrt(rho/(1-rho))*norm.pdf((C(g,s)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        Term2 = np.sqrt(rho/(1-rho))*norm.pdf((C(g,s-1)-x*np.sqrt(rho))/(np.sqrt(1-rho))) 
        return Term1 + Term2
    
    def pi_prime_prime(x,s):
        Term1 = -(rho/(1-rho))*((C(g,s)-x*np.sqrt(rho))/(np.sqrt(1-rho)))*norm.pdf((C(g,s)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        Term2 = (rho/(1-rho))*((C(g,s-1)-x*np.sqrt(rho))/(np.sqrt(1-rho)))*norm.pdf((C(g,s-1)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        return Term1 + Term2
    
    def pi_prime_prime_prime(x,s):
        Term1 = ((rho/(1-rho))**(3/2))*(1-((C(g,s)-x*np.sqrt(rho))/(np.sqrt(1-rho)))**2)*norm.pdf((C(g,s)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        Term2 = ((rho/(1-rho))**(3/2))*(((C(g,s-1)-x*np.sqrt(rho))/(np.sqrt(1-rho)))**2-1)*norm.pdf((C(g,s-1)-x*np.sqrt(rho))/(np.sqrt(1-rho)))
        return Term1 + Term2
    
    def mu(x):
        return np.sum([lambda_(s)*pi_(x,s) for s in range(S+1)],0)   
    
    def mu_prime(x):   
        return np.sum([lambda_(s)*pi_prime(x,s) for s in range(S+1)],0)
    
    def mu_prime_prime(x):
        return np.sum([lambda_(s)*pi_prime_prime(x,s) for s in range(S+1)],0)
    
    def mu_prime_prime_prime(x):
        return np.sum([lambda_(s)*pi_prime_prime_prime(x,s) for s in range(S+1)],0)
    
    def sigma_n_square(x):
        Term1 = (xi_0squared()+lambda_(0)**2)*pi_(x,0)
        Term2 = np.sum([(lambda_(s)**2)*pi_(x,s) for s in range(1,S+1)],0)
        Term3 = -mu(x)**2    
        return np.array(Term1) + np.array(Term2) + np.array(Term3)
    
    def sigma_n_square_prime(x):
        Term1 = (xi_0squared()+lambda_(0)**2)*pi_prime(x,0)
        Term2 = np.sum([(lambda_(s)**2)*pi_prime(x,s) for s in range(1,S+1)],0)
        Term3 = -2*mu(x)*mu_prime(x)       
        return np.array(Term1) + np.array(Term2) + np.array(Term3)
    
    def sigma_n_square_prime_prime(x):
        Term1 = (xi_0squared()+lambda_(0)**2)*pi_prime_prime(x,0)
        Term2 = np.sum([(lambda_(s)**2)*pi_prime_prime(x,s) for s in range(1,S+1)],0)
        Term3 = -2*(mu_prime(x)**2+mu_prime_prime(x)*mu(x))
        return np.array(Term1) + np.array(Term2) + np.array(Term3)
    
    
    numerator1 = np.sum((A**2)*sigma_n_square(alpha_X))
    denominator1 = np.sum(A*mu_prime(alpha_X))
    Summand1 = -alpha_X*((numerator1)/(denominator1))
    
    numerator2 = np.sum((A**2)*sigma_n_square_prime(alpha_X))
    denominator2 = np.sum(A*mu_prime(alpha_X))
    Summand2 = (numerator2)/(denominator2)

    numerator3 = (np.sum((A**2)*sigma_n_square(alpha_X))*(np.sum(A*mu_prime_prime(alpha_X))))
    denominator3 = (np.sum(A*mu_prime(alpha_X)))**2
    Summand3 = -(numerator3)/(denominator3)
            
    GA_1st = 0.5*np.exp(-r(T))*(Summand1+Summand2+Summand3)
    if second_order == False:       
        GA_2nd = 0
    else:
        central_third = -(2*(1-2*ELGD)/(1+1/nu))*VLGD_squared()
        if  LGD_constant:
            central_third = 0 # Under the assumption ELGD^3 = E[LGD^3]
        
        eta_0 = (F(T)/(P_00))**3 *(central_third)
        
        def eta(x):
            Term1 = (eta_0+3*(xi_0squared()+lambda_(0)**2)*lambda_(0) -2*lambda_(0)**3)*pi_(x,0)
            Term2 = np.sum([(lambda_(s)**3)*pi_(x,s) for s in range(1,S+1)],0)
            Term3 = -3*(sigma_n_square(x)+mu(x)**2)*mu(x)+2*mu(x)**3
            return Term1+Term2+Term3
        
        def eta_prime(x):
            Term1 = (eta_0+3*(xi_0squared()+lambda_(0)**2)*lambda_(0) -2*lambda_(0)**3)*pi_prime(x,0)
            Term2 = np.sum([(lambda_(s)**3)*pi_prime(x,s) for s in range(1,S+1)],0)
            Term3 = -3*mu(x)*(sigma_n_square_prime(x)+2*mu_prime(x)*mu(x))
            Term4 = 3*mu_prime(x)*(mu(x)**2-sigma_n_square(x))             
            return Term1+Term2+Term3+Term4
        
        def eta_prime_prime(x):
            Term1 = (eta_0+3*(xi_0squared()+lambda_(0)**2)*lambda_(0) -2*lambda_(0)**3)*pi_prime_prime(x,0)
            Term2 = np.sum([(lambda_(s)**3)*pi_prime_prime(x,s) for s in range(1,S+1)],0)
            Term3 = -3*mu(x)*(sigma_n_square_prime_prime(x)+2*mu_prime_prime(x)*mu(x)+2*(mu_prime(x)**2))
            Term4 = -6*mu_prime(x)*sigma_n_square_prime(x)
            Term5 = 3*mu_prime_prime(x)*(mu(x)**2-sigma_n_square(x))   
            return Term1+Term2+Term3+Term4+Term5
        
        m1_prime = -np.sum(A*mu_prime(alpha_X))*np.exp(-r(T))
        m1_prime_prime = -np.sum(A*mu_prime_prime(alpha_X))*np.exp(-r(T))
        m1_prime_prime_prime = -np.sum(A*mu_prime_prime_prime(alpha_X))*np.exp(-r(T))
                
        m3 = -np.exp(-3*r(T))*np.sum(eta(alpha_X)*(A**3))
        m3_prime =  -np.exp(-3*r(T))*np.sum(eta_prime(alpha_X)*(A**3))
        m3_prime_prime =  -np.exp(-3*r(T))*np.sum(eta_prime_prime(alpha_X)*(A**3))
        
        h_prime_over_h = -alpha_X
        h_prime_prime_over_h = (alpha_X**2-1)
        
        Factor1 = 1/(6*(m1_prime**2))
        Summand1 = m3_prime_prime
        Summand2 = -m3_prime*(-2*h_prime_over_h+3*(m1_prime_prime/m1_prime))
        Factor2 = m3
        Summand3 = h_prime_prime_over_h
        Summand4 = -3*h_prime_over_h*(m1_prime_prime)/m1_prime
        Summand5 = (3*(m1_prime_prime)**2-m1_prime*m1_prime_prime_prime)/(m1_prime**2) 
        GA_2nd =  Factor1*(Summand1+Summand2+Factor2*(Summand3+Summand4+Summand5))
    return GA_1st+GA_2nd

def GA_Sim_CM(PD, # Vector of default probabillities
          ELGD, #Vector of expected losses given default,
          A, # Vector of Exposures at default,
          M, # Vector of Maturities
          q, # quantile level          
         r, # interest rate
         T, # Maturity
         g, # current states
         trans_prob,#N times S matrix,
         psi = 0.4, # Market Sharpe Ratio,
         rho =0.2,
         nu = 0.1,
         c = 0.05, #Coupons of bonds
         S = 10, # Number of States of Rating (A and B),
         N_simulations = 10000,
         epsilon_scenarios = 0.2,
         LGD_constant = False,
         load_samples = True,
         sample_file = "samples.csv",
         default_only = False):
    
    
    
    N = len(A)  # Number of obligors
    threshold = 2**10# Instead of Inf for function C
    alpha_X = norm.ppf(1-q) # q-Quantile of X
    
    
    tenor = [np.array([i*0.5 for i in range((int(M[j]))*2+1)]) for j in range(N)]
    tenor = np.array(tenor,dtype = object)
    
    
    M = np.round(M*2,0)/2 # round to 0.5 digits
    
    #Convert to numpy-arrays
    PD, ELGD, A, M, g = np.array(PD), np.array(ELGD), np.array(A), np.array(M), np.array(g)
    
    #Create a matrix corresponding to g
    g_matrix = np.zeros((N,S+1))
    for i in range(N):
        g_matrix[i,g[i]] = 1
    
    # Normalize exposures, and consider weights
    A = A/np.sum(A)
    
    def convert_matrix_order(matrix_0):  # Function to rearrange the transition matrix s.t. s=0 corresponds to default
        reverse_indices = list(matrix_0.columns)[::-1]
        matrix_0 = matrix_0.reindex(columns=reverse_indices)
        matrix_0 = matrix_0.reindex(reverse_indices)
        return matrix_0
    
    if default_only == True: # No transitions allowed
        matrix_0 = convert_matrix_order(trans_prob)
        indices = matrix_0.columns
        sums = matrix_0.apply(lambda x: np.sum(x[1:]),1)
        converted_trans_prob = pd.DataFrame(np.diag(sums), index = indices, columns = indices)
        converted_trans_prob['D'] = matrix_0['D']
    else:        
        # Change the order of the columns and rows of the transition matrix
        converted_trans_prob = convert_matrix_order(trans_prob) #matrix where s=0: Default ~ s=1:B ~ s=2: A
    converted_trans_prob_half_yr = sqrtm(converted_trans_prob)
 
    
    
    def C(g,s): # Determination of bounds for default, g vector, s scalar
        eps = 1e-10         
        if s == S:
            return np.array([threshold]*N)
        if s == -1:
            return np.array([-threshold]*N)     
        else:
            val = np.sum(np.matmul(g_matrix,converted_trans_prob.iloc[:S+1,:s+1]),1)
            val = np.maximum(np.minimum(val,1-eps),eps)
            val = norm.ppf(val) 
            val = np.maximum(np.minimum(val, np.array([threshold]*N)),np.array([-threshold]*N))
        return val
        
        
    

    def p(s,t1,t2): #probability of default between t and T
        if isinstance(s, int): # if s is an integer number
            if t1 == t2:
                if s == 0:
                    return 1
                else:
                    return 0
            else:
                state = np.zeros(S+1)
                state[s] = 1 # indicate state s
                result = np.matmul(state,np.array(converted_trans_prob_half_yr))
                for i in range(max(int((t2-t1-0.5)*2),0)): 
                    result = np.matmul(result,np.array(converted_trans_prob_half_yr))
                return np.maximum(result[0],0)
                
        else: # if s is a vector
            if t1 == t2:
                return np.array([int(s[i]==0) for i in range(len(s))])
            else:
                state = np.zeros((len(s),S+1))
                for i in range(len(s)):
                    state[i,s[i]] = 1 # indicate state s
                result = np.matmul(state,np.array(converted_trans_prob_half_yr))
                for i in range(max(int((t2-t1-0.5)*2),0)):  
                    result = np.matmul(result,np.array(converted_trans_prob_half_yr))
                return np.maximum(result[:,0],0)    
            
            
    def ps_star(s,t1,t2):
        return norm.cdf(norm.ppf(p(s,t1,t2))+psi*np.sqrt(t2-t1)*np.sqrt(rho))  
        
    # def F(t):  #Forward value of the bond
    #     #Eva
    #     Term1 =c*0.5*np.sum(np.exp(-r*(t-tenor[:,int(t*2):])),1)
    #     Term2 = np.exp(-r*(tenor[:,-1]-t))
    #     return Term1 +Term2 
   
    def F_vector(t):  #Forward value of the bond
        #Eva
        Term1 =  c*0.5*np.array([np.sum([np.exp(-r(t_i-t)) for t_i in tenor[j][1:] if t_i >= t]) for j in range(N)])
        Term2 = np.array([np.exp(-r(tenor[j][-1]-t)) for j in range(N)])
        return Term1 + Term2
    
    # Compute once the F values
    tenor_list = [i*0.5 for i in range((int(np.max(M)))*2+1)]
    F_vec = [F_vector(t) for t in tenor_list]
    
    def F(t):
        return F_vec[int(np.minimum(2*t,np.max(tenor_list)))]
    
        
    def P_0(LGD=ELGD):
        Term1 = F(0)
        # Eva:
        Term2 = -np.array([np.sum([np.exp(-r(tenor[j][i]))*(ps_star(g,0,tenor[j][i])[j] \
                        -ps_star(g,0,tenor[j][i-1])[j])*(F(tenor[j][i])[j]*LGD[j]) for i in range(1,len(tenor[j]))]) for j in range(N)])
        # Gordy:
        #Term2 = -np.sum([np.exp(-r*(tenor[i]))*(ps_star(g,0,tenor[i]) \
        #                -ps_star(g,0,tenor[i-1]))*(F(tenor[i])-(1-ELGD)*(1+c/2)) for i in range(1,len(tenor))],0)
        val = Term1 + Term2
        return val #Bond price for each obligor (N-dim vector)
    
    # Compute the value P_0 once.
    P_00 = P_0() 
    
    def lambda_vector(s,LGD=ELGD):
        if s >0:
            #Eva:
            Term1 = c*0.5*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T]) for j in range(N)])
            Term2 = F(T)
            Term3 = - np.array([np.sum([np.exp(-r(tenor[j][i]-T))*(ps_star(s,0,tenor[j][i]-T)[j] \
                            -ps_star(s,0,tenor[j][i-1]-T)[j])*(LGD[j])*F(tenor[j][i])[j] for i in range(1,len(tenor[j])) if tenor[j][i]>T]) for j in range(N)])
            #Gordy
            #Term1 = c*0.5*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <=T])
            #Term2 = F(T)
            #Term3 = np.sum([np.exp(-r*(tenor[i]-T))*(ps_star(s,0,tenor[i]-T) \
            #            -ps_star(s,0,tenor[i-1]-T))*(F(tenor[i])-(1-ELGD)*(1+c/2)) for i in range(1,len(tenor)) if tenor[i]>T],0)
    
            lam = (1/ P_0(LGD))*(Term1 + Term2 +Term3)           
    
        elif s==0:
            if LGD_constant and default_only:
                lam = (1-ELGD)
            else:
                #Eva:
                lam = (1/ P_00)*((c/2)*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T])for j in range(N)])+(1-LGD)*F(T)) 
            #Gordy:
            #lam = (1/ P_0())*((c/2)*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <T])+(1-ELGD)*(1+c/2))
        return lam
    
    lambda_vec = [lambda_vector(s) for s in range(S+1)]
    
                  
    
    def lambda_(s):
        return lambda_vec[s]
    
        
    
            
     
    VLGD =  nu*(ELGD)*(1-ELGD)
    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta
    alpha, beta = parameters_beta(ELGD,VLGD)
    

    
    def determine_rating(Y): # Determine ratings in dependence of realization of Y        
        one_vector = [(Y>=rating_bounds[i])*(Y<=rating_bounds[i+1]) for i in range(len(rating_bounds)-1)]
        return np.matrix(one_vector)
    
    if load_samples:
        # Determine Rating Bounds
        rating_bounds = [C(g,s) for s in range(-1,S+1)]
        # Expected Return
        ER = np.sum([lambda_(s)*np.array(np.matmul(g_matrix,converted_trans_prob).iloc[:,s])*A for s in range(S+1)])
        # Read the samples
        samples = pd.read_csv(sample_file)
        # The sampled LGDs
        LGD = samples[["LGD"+str(s+1) for s in range(N)]]
        #The sampled Y's
        Y = samples[["Y"+str(s+1) for s in range(N)]]
        # Compute the return R, Vectorize!
        R = np.array([np.sum(np.multiply(np.sum(np.multiply(np.matrix([lambda_(s,np.array(LGD.iloc[i].to_numpy())) for s in range(S+1)]),determine_rating(Y.iloc[i].to_numpy()) ),0),A))  for i in range(len(LGD))])
        # Compute the loss
        L = (ER-R)*np.exp(-r(T)*T)
        # Compute Quantiles
        df = pd.DataFrame({
        'L': L,
        'X': samples["X"]})
        df = (df.sort_values(by=['X']))
        quantile_loss_L = np.quantile(L,q)
        quantile_loss_X = np.quantile(samples["X"],1-q)
        conditional_loss = np.mean(df['L'][np.abs(df['X']-quantile_loss_X)<epsilon_scenarios])
        
    elif load_samples == False: 
        # Expected Return
        ER = np.sum([lambda_(s)*np.array(np.matmul(g_matrix,converted_trans_prob).iloc[:,s])*A for s in range(S+1)])
        L_list = []
        X_list = []
        rating_bounds = [C(g,s) for s in range(-1,S+1)]
        for i in range(N_simulations):
            X, eps  = np.random.normal(size = 1), np.random.normal(size = N)
            Y = np.sqrt(rho)*X+np.sqrt(1-rho)*eps
            #Determine rating classes at horizon T=1:
            ratings = determine_rating(Y)     
            # Sample LGD
            #LGD = np.random.beta(alpha,beta,size = N)
            #LGD = LGD_constant*ELGD + (1-LGD_constant)*LGD
            # Compute realized returns based on the states (depending on Y) and based on LGD
            #LGD = np.random.beta(alpha,beta,size = N)
            R = np.sum(np.multiply(np.sum(np.multiply(np.matrix([lambda_(s) for s in range(S+1)]),ratings),0),A))
            loss = (ER-R)*np.exp(-r(T)*T)
            L_list.append(loss)
            X_list.append(X)
        quantile_loss_L = np.quantile(L_list,q)
        quantile_loss_X = np.quantile(X_list,1-q)
        
        df = pd.DataFrame({
        'L': L_list,
        'X': X_list})
        df = (df.sort_values(by=['X']))
        conditional_loss = np.mean(df['L'][np.abs(df['X']-quantile_loss_X)<epsilon_scenarios])
        #print("No of relevant scenarios: {}".format(len(df['L'][np.abs(df['X']-quantile_loss_X)<epsilon_scenarios])))
    return quantile_loss_L-conditional_loss



def GA_Sim(PD, ELGD, A, T, q, rho = 0.2,
           r = 0.0,N_simulations = 10000,epsilon_scenarios = 0.2,nu =0.1,
           LGD_constant = False,load_samples = True,
         sample_file = "samples.csv"):
    PD, ELGD, A, rho = np.array(PD), np.array(ELGD), np.array(A), np.array(rho)        
    VLGD =  nu*(ELGD)*(1-ELGD)
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta
    alpha, beta = parameters_beta(ELGD,VLGD)
    A = A/np.sum(A)
    N = len(ELGD)
    
    if load_samples:
        samples = pd.read_csv(sample_file)
        # The sampled LGDs
        LGD = samples[["LGD"+str(s+1) for s in range(N)]]
        #The sampled Y's
        R = samples[["Y"+str(s+1) for s in range(N)]]
        ER = np.sum(A*ELGD*(PD))
        probs = norm.ppf(PD)
        L = (np.sum(np.tile(A,[len(LGD),1])*LGD.values*R.lt(probs.squeeze(),axis = 1).values,1)-ER)*np.exp(-r*T)
        
        quantile_loss_L = np.quantile(L,q)
        quantile_loss_X = np.quantile(samples["X"],1-q)
        df = pd.DataFrame({
        'L': L,
        'X': samples["X"]})
        df = (df.sort_values(by=['X']))
        conditional_loss = np.mean(df['L'][np.abs(df['X']-quantile_loss_X)<epsilon_scenarios])    
    elif load_samples == False:
        L_list = []
        X_list = []
        for i in range(N_simulations):
            X = np.random.normal(size = 1)
            eps = np.random.normal(size = N)
            R = np.sqrt(rho)*X+np.sqrt(1-rho)*eps
            LGD = np.random.beta(alpha,beta,size = N)
            LGD = LGD_constant*ELGD + (1-LGD_constant)*LGD 

            loss = (np.sum(A*LGD*(R<norm.ppf(PD)))-np.sum(A*ELGD*(PD)))*np.exp(-r*T)
            L_list.append(loss)
            X_list.append(X)
        quantile_loss_L = np.quantile(L_list,q)
        quantile_loss_X = np.quantile(X_list,1-q)
        df = pd.DataFrame({
        'L': L_list,
        'X': X_list})
        df = (df.sort_values(by=['X']))
        conditional_loss = np.mean(df['L'][np.abs(df['X']-quantile_loss_X)<epsilon_scenarios])
    #print("No of relevant scenarios: {}".format(len(df['L'][np.abs(df['X']-quantile_loss_X)<epsilon_scenarios])))
    return quantile_loss_L-conditional_loss

# Function that determines the tenor list
def tenor_list(years):
    return [i*0.5 for i in range(years*2+1)] # from 0 until x years, every 6 months

# Function to create Scenarios
def create_scenarios(ELGD, N_simulations,  nu = 0.1, rho = 0.2, filename = "samples.csv"):
    ELGD = np.array(ELGD)
    N = len(ELGD) # Number of obligors
    VLGD =  nu*(ELGD)*(1-ELGD) # Variance of LGD
    
    # Determine Parameters for the LGD variables    
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta
    
    alpha, beta = parameters_beta(ELGD,VLGD)
        
    MC_list = []
        
    for i in range(N_simulations):
        X, eps  = np.random.normal(size = 1), np.random.normal(size = N)
        Y = np.sqrt(rho)*X+np.sqrt(1-rho)*eps
        LGD = np.random.beta(alpha,beta,size = N)
        MC_list.append(X.tolist()+Y.tolist()+LGD.tolist())
        
    header_string = ["X"]+["Y"+str(s+1) for s in range(N)]+["LGD"+str(s+1) for s in range(N)]
    df = pd.DataFrame(MC_list, columns = header_string)
    df.to_csv(filename, index = False, header=True)   