import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
from scipy.linalg import sqrtm
from scipy.optimize import fsolve

def create_sample_actuarial(ELGD_bounds,EAD_bounds,PD_bounds,omega_bounds,obligor_bounds,B,
                            PDs = 0,PDs_Sample = False):
    if PDs_Sample:
        PD = np.random.choice(PDs,size = (B,obligor_bounds[1]))
        ELGD = np.random.uniform(ELGD_bounds[0],ELGD_bounds[1],(B,obligor_bounds[1]))
        EAD = np.random.uniform(EAD_bounds[0],EAD_bounds[1],(B,obligor_bounds[1]))
        omega = np.random.uniform(omega_bounds[0],omega_bounds[1],(B,obligor_bounds[1]))
    else:
        ELGD = np.random.uniform(ELGD_bounds[0],ELGD_bounds[1],(B,obligor_bounds[1]))
        EAD = np.random.uniform(EAD_bounds[0],EAD_bounds[1],(B,obligor_bounds[1]))
        PD = np.random.uniform(PD_bounds[0],PD_bounds[1],(B,obligor_bounds[1]))    
        omega = np.random.uniform(omega_bounds[0],omega_bounds[1],(B,obligor_bounds[1]))
    # Create vector with number of obligors
    N_obligors = np.random.randint(obligor_bounds[0], obligor_bounds[1],size = B)
    N = np.zeros((B,obligor_bounds[1]))
    for i in range(B):
        N[i,:] = np.pad(np.repeat(1,N_obligors[i]),(0,obligor_bounds[1]-N_obligors[i])) # Number of Obligors    
    return ELGD*N, EAD*N / np.sum(EAD*N) ,PD*N,omega*N



def MC_IS_actuarial(ELGD,EAD,PD,omega= 0.5,q=0.999,n = 10000,n1 = 1,xi = 0.25,IRB = False,
                  constant_rho = True,
                  constant_rho_val = 0.35,
                  LGD_constant = False,
                  nu = 0.25,
                  M = 1):
    
    eps = 1e-10
    b = (0.11852-0.05478*np.log(np.maximum(PD,eps)))**2 # Maturity adjustment
    MA = (1/(1-1.5*b))*(1+(M-2.5)*b)
    # Define  Number of obligors and Batch Size
    n_obligors = EAD.shape[1]
    B = EAD.shape[0]
    # Normalize EADs
    EAD = EAD / np.repeat(np.expand_dims(np.sum(EAD,1),1),n_obligors,axis = 1)
    # Expected Loss
    #EL = np.sum(ELGD*EAD*PD,1)
    quantile = gamma.ppf(q=q,a = xi,scale = 1/xi)
    #y = EL*quantile/3
    if IRB:
        def determine_omega(om):
            #b = (0.11852-0.05478*np.log(PD))**2
            R_corr = 0.12*(1-np.exp(-50*PD))/(1-np.exp(-50))+0.24*(1-(1-np.exp(-50*PD))/(1-np.exp(-50)))
            K = ((ELGD)*norm.cdf(np.sqrt(1/(1-R_corr))*norm.ppf(PD)+np.sqrt(R_corr/(1-R_corr))*norm.ppf(q))-PD*ELGD)
            alpha_X =  gamma.ppf(q,a =0.25, scale = 1/0.25)
            return  np.reshape((np.abs(ELGD*PD*om*(alpha_X-1)-K)),n_obligors)
        omega = np.reshape(fsolve(determine_omega,[0.25]*n_obligors),(B,n_obligors))
        omega[EAD==0] = 0
    if constant_rho:
        def determine_omega(om):
            #b = (0.11852-0.05478*np.log(PD))**2
            R_corr = np.array([constant_rho_val]*len(PD))
            K = ((ELGD)*norm.cdf(np.sqrt(1/(1-R_corr))*norm.ppf(PD)+np.sqrt(R_corr/(1-R_corr))*norm.ppf(q))-PD*ELGD)
            alpha_X =  gamma.ppf(q,a =0.25, scale = 1/0.25)
            return  np.reshape((np.abs(ELGD*PD*om*(alpha_X-1)-K)),n_obligors)
        omega = np.reshape(fsolve(determine_omega,[0.25]*n_obligors),(B,n_obligors))
        omega[EAD==0] = 0        
        #omega = fsolve(to_solve,np.reshape([0.25]*n_obligors,(B,n_obligors)))
    def extend_dim(vec):
        return np.repeat(np.expand_dims(vec,1),n_obligors,axis = 1)
    
    # def to_derive(T):
    #     return np.sum((PD*(1-omega)*(np.exp(extend_dim(T)*ELGD*EAD)-1)),1)-xi*np.log(1-(1/xi)*np.sum(PD*omega*(np.exp(extend_dim(T)*ELGD*EAD)-1),1))
    # def to_solve(T,eps = 1e-2):
    #     #return (to_derive(T+eps)-to_derive(T))/eps-y 
    #     sum1 = np.sum(PD*(1-omega)*ELGD*EAD*(np.exp(extend_dim(T)*ELGD*EAD)),1)
    #     sum2 = -np.sum(PD*omega*ELGD*EAD*np.exp(extend_dim(T)*ELGD*EAD),1)/((1-(1/xi)*np.sum(PD*omega*(np.exp(extend_dim(T)*ELGD*EAD)-1),1)))
    #     return sum1-sum2-y    
    
    def to_solve_2(T,eps = 1e-3):
        return np.sum(PD*omega*(np.exp(EAD*ELGD*extend_dim(T))-1),1)-xi*(1-1/(quantile/2))
    
    # Determination of t and T s.t. mean of X equals quantile
    T = fsolve(to_solve_2,np.array([10]*B),full_output = False)
    #T = fsolve(to_solve,np.array([10]*B),full_output = False)
    #print(T)
    T_extended = extend_dim(T)
    t = np.sum(PD*omega*(np.exp(EAD*ELGD*T_extended)-1),1)
    
    def r(x):
        return np.exp(-t*x-xi*np.log(1-(1/xi)*t))
    
    def r_tilde(x,l):
        sum1 = np.exp(EAD*ELGD*T_extended)*np.maximum(np.minimum(PD*(1+omega*(x-1)),1),0)
        sum2 = np.maximum(np.minimum(PD*(1+omega*(x-1)),1),0)
        return np.exp(-T*l)*np.prod(sum1+1-sum2,1)
    
    
    def r_tilde(x,l):
        return np.exp(-T*l)*np.prod(np.exp(EAD*ELGD*T_extended)*np.minimum(PD*(1+omega*(x-1)),1)+1-np.minimum(PD*(1+omega*(x-1)),1),1)
                                     
    def r_total(l):
        return np.exp(-T*l+np.sum(PD*(1-omega)*(np.exp(EAD*ELGD*T_extended)-1),1)-xi*np.log(1-(1/xi)*t))
    
    
    L = []
    #L_2 = []
    p_list = []
    #p_list_2 = []
    
    X = np.array([np.random.gamma(shape = xi,scale = (1/xi)/(1-(1/xi)*t[i]), size = n) for i in range(B)])
    def parameters_beta(mu, var): # Convert mean, variance to alpha, beta
        alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return alpha, beta       
    if LGD_constant == False:
        alpha_2, beta_2 = parameters_beta(ELGD[ELGD>0],nu*(ELGD[ELGD>0])*(1-ELGD[ELGD>0]))
        LGD = np.nan_to_num(np.array([[np.pad(np.random.beta(alpha_2,beta_2,size = (B,np.sum(ELGD>0))),((0,0),(0,n_obligors-np.sum(ELGD>0))))
                                       for _ in range(n1)] for _ in range(n)])) #(n,n1, B,n_obligors)
    else:
        LGD = np.tile(ELGD[np.newaxis,np.newaxis,:],(n,n1,1,1))
    # The Monte Carlo Loop for the Conditional Expectation and the Quantile
    for i in range(n):        
        pi_X = np.maximum(np.minimum(PD*(1+omega*(extend_dim(X[:,i])-1)),1),0)
        L_inner = []
        #L_inner_2 = []
        p_inner = []
        for j in range(n1):
            D = np.random.binomial(1,np.minimum((np.exp(T_extended*EAD*ELGD)*pi_X)/(np.exp(T_extended*EAD*ELGD)*pi_X+1-pi_X),1))
            loss_var = np.sum(MA*EAD*LGD[i,j,:]*D,axis = 1)
            L_inner.append(loss_var)
            #L_inner_2.append(loss_var)
            p_inner.append(r_tilde(extend_dim(X[:,i]),loss_var))
        L.append(np.mean(L_inner,axis =0))
        #L_2.append(np.mean(L_inner_2,axis =0))
        p_list.append(r(X[:,i])*np.mean(p_inner,0))
        #p_list_2.append( r(X[:,i]))
        

    # Quantile Computation
    MC_quantiles = []
    for b in range(B):
        sorted_data = sorted(zip([L[i][b] for i in range(len(L))], [p_list[i][b] for i in range(len(p_list))])) # Sort the by L
        sorted_L, sorted_p = zip(*sorted_data)
        quantile_index = np.searchsorted(np.cumsum(sorted_p), q*np.sum([p_list[i][b] for i in range(len(p_list))]), side='right')
        MC_quantiles.append(sorted_L[quantile_index-1])
    
    
    
    # Conditional Expectations
    eps = quantile*0.2
    express_1 = np.sum((np.array(p_list)*np.array(L)).transpose()*(np.abs(X-quantile)<eps),axis = 1)
    express_2 = np.maximum(np.sum(np.array(p_list).transpose()*(np.abs(X-quantile)<eps),axis =1),0.00000000000001)
    MC_cond_exp = express_1/express_2

    #print(np.array(MC_quantiles)-MC_cond_exp)
    return np.array(MC_quantiles)-MC_cond_exp