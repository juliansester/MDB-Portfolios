import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.optimize import fsolve
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve

#def r_nelson(x): # Estimate from 2022
#    return NelsonSiegelSvenssonCurve(1.437506, -1.928506, 13.008597, -14.402321, 1.089999, 1.090450)(x)/100
def r_nelson(x): # Estimate from 2022
    return NelsonSiegelSvenssonCurve(3.95812666292673e-06, 5.05669120514791, -169.651342441383, 175.470340786573, 9.10233964222207, 9.54062325103228)(x)/100
			 	

S = 17 # Number of total states
# First Define a Function that reads a transition matrix:
def trans_matrix_to_dict(matrix,S = 17):
    sp_transition = matrix
    transition_matrix_entries = {'Gs1': list(sp_transition['AAA'][-18:-1])+[0.],
                        'Gs2': list(sp_transition['AA+'][-18:-1])+[0.],
                        'Gs3': list(sp_transition['AA'][-18:-1])+[0.],
                        'Gs4': list(sp_transition['AA-'][-18:-1])+[0.],
                        'Gs5':  list(sp_transition['A+'][-18:-1])+[0.],
                          'Gs6': list(sp_transition['A'][-18:-1])+[0.],
                        'Gs7':list(sp_transition['A-'][-18:-1])+[0.],
                        'Gs8': list(sp_transition['BBB+'][-18:-1])+[0.],
                          'Gs9': list(sp_transition['BBB'][-18:-1])+[0.],
                          'Gs10': list(sp_transition['BBB-'][-18:-1])+[0.],
                          'Gs11': list(sp_transition['BB+'][-18:-1])+[0.],
                          'Gs12': list(sp_transition['BB'][-18:-1])+[0.],
                          'Gs13': list(sp_transition['BB-'][-18:-1])+[0.],
                            'Gs14': list(sp_transition['B+'][-18:-1])+[0.],
                            'Gs15': list(sp_transition['B'][-18:-1])+[0.],
                            'Gs16': list(sp_transition['B-'][-18:-1])+[0.],
                           'Gs17': list(sp_transition['Cs'][-18:-1])+[0.],
                        'D': list(sp_transition['D'][-18:-1])+[100.0]}
    return_dict = {}
    
    trans_prob =  pd.DataFrame(transition_matrix_entries, index = ["Gs1","Gs2","Gs3","Gs4","Gs5",
                                                                        "Gs6","Gs7","Gs8","Gs9",
                                                                        "Gs10",'Gs11','Gs12','Gs13',
                                                                        "Gs14",'Gs15','Gs16','Gs17',"D"])/100

    trans_prob = trans_prob/ np.sum(trans_prob,1)
    def convert_matrix_order(matrix_0,default_only=False):  # Function to rearrange the transition matrix s.t. s=0 corresponds to default
        reverse_indices = list(matrix_0.columns)[::-1]
        matrix_0 = matrix_0.reindex(columns=reverse_indices)
        matrix_0 = matrix_0.reindex(reverse_indices)
        if default_only == True: # No transitions allowed
            indices = matrix_0.columns
            sums = matrix_0.apply(lambda x: np.sum(x[1:]),1)
            matrix_1 = pd.DataFrame(np.diag(sums), index = indices, columns = indices)
            matrix_1['D'] = matrix_0['D']
            return matrix_1
        return matrix_0
        

    # else:        
        # Change the order of the columns and rows of the transition matrix
    converted_trans_prob = convert_matrix_order(trans_prob) #matrix where s=0: Default ~ s=1:B ~ s=2: A
    converted_trans_prob_do = convert_matrix_order(trans_prob,default_only=True) #matrix where s=0: Default ~ s=1:B ~ s=2: A
    converted_trans_prob_half_yr = sqrtm(converted_trans_prob)
    converted_trans_prob_half_yr_do = sqrtm(converted_trans_prob_do)

    return_dict["trans_prob"] = trans_prob
    return_dict["converted_trans_prob"] = converted_trans_prob
    return_dict["converted_trans_prob_do"] = converted_trans_prob_do
    return_dict["converted_trans_prob_half_yr"] = converted_trans_prob_half_yr
    return_dict["converted_trans_prob_half_yr_do"] = converted_trans_prob_half_yr_do
    return return_dict




def C(ELGD,A,rho,c,g,D,s,trans_dict,default_only=False,threshold = 2**10, eps = 1e-10): # Determination of bounds for default, g vector, s scalar
    N = len(A)  # Number of obligors
    #Create a matrix corresponding to g
    g_matrix = np.zeros((N,S+1))
    for i in range(N):
        g_matrix[i,g[i]] = 1
   
    if s == S:
        return np.array([threshold]*N)
    if s == -1:
        return np.array([-threshold]*N)     
    else:
        if default_only:
            val = np.sum(np.matmul(g_matrix,trans_dict["converted_trans_prob_do"].iloc[:S+1,:s+1]),1)
            val = np.maximum(np.minimum(val,1-eps),eps)
            val = norm.ppf(val) 
            val = np.maximum(np.minimum(val, np.array([threshold]*N)),np.array([-threshold]*N))
        else:
            val = np.sum(np.matmul(g_matrix,trans_dict["converted_trans_prob"].iloc[:S+1,:s+1]),1)
            val = np.maximum(np.minimum(val,1-eps),eps)
            val = norm.ppf(val) 
            val = np.maximum(np.minimum(val, np.array([threshold]*N)),np.array([-threshold]*N))
    return val


def p(s,t1,t2,trans_dict,default_only = False): #probability of default between t and T
    if default_only:
        if isinstance(s, int): # if s is an integer number
            if t1 == t2:
                if s == 0:
                    return 1
                else:
                    return 0
            else:
                state = np.zeros(S+1)
                state[s] = 1 # indicate state s
                result = np.matmul(state,np.array(trans_dict["converted_trans_prob_half_yr_do"]))
                for i in range(max(int((t2-t1-0.5)*2),0)): 
                    result = np.matmul(result,np.array(trans_dict["converted_trans_prob_half_yr_do"]))
                return np.maximum(result[0],0)
                
        else: # if s is a vector
            if t1 == t2:
                return np.array([int(s[i]==0) for i in range(len(s))])
            else:
                state = np.zeros((len(s),S+1))
                for i in range(len(s)):
                    state[i,s[i]] = 1 # indicate state s
                result = np.matmul(state,np.array(trans_dict["converted_trans_prob_half_yr_do"]))
                for i in range(max(int((t2-t1-0.5)*2),0)):  
                    result = np.matmul(result,np.array(trans_dict["converted_trans_prob_half_yr_do"]))
                return np.maximum(result[:,0],0) 
    elif default_only ==False:
        if isinstance(s, int): # if s is an integer number
            if t1 == t2:
                if s == 0:
                    return 1
                else:
                    return 0
            else:
                state = np.zeros(S+1)
                state[s] = 1 # indicate state s
                result = np.matmul(state,np.array(trans_dict["converted_trans_prob_half_yr"]))
                for i in range(max(int((t2-t1-0.5)*2),0)): 
                    result = np.matmul(result,np.array(trans_dict["converted_trans_prob_half_yr"]))
                return np.maximum(result[0],0)
                
        else: # if s is a vector
            if t1 == t2:
                return np.array([int(s[i]==0) for i in range(len(s))])
            else:
                state = np.zeros((len(s),S+1))
                for i in range(len(s)):
                    state[i,s[i]] = 1 # indicate state s
                result = np.matmul(state,np.array(trans_dict["converted_trans_prob_half_yr"]))
                for i in range(max(int((t2-t1-0.5)*2),0)):  
                    result = np.matmul(result,np.array(trans_dict["converted_trans_prob_half_yr"]))
                return np.maximum(result[:,0],0)         
            
            
def ps_star(s,t1,t2,rho,trans_dict,psi =0.4,default_only=False,eps = 1e-20):
    prob = np.minimum(np.maximum(p(s,t1,t2,trans_dict,default_only=default_only),eps),1-eps)
    return norm.cdf(norm.ppf(prob)+psi*np.sqrt(np.maximum(t2-t1,0))*np.sqrt(rho))  
    
def determine_rating(ELGD,A,rho,c,g,D,Y,trans_dict,default_only=False): # Determine ratings in dependence of realization of Y
    rating_bounds_vec = [C(ELGD,A,rho,c,g,D,s,trans_dict,default_only) for s in range(-1,S+1)] 
    one_vector = [(Y>=rating_bounds_vec[i])*(Y<=rating_bounds_vec[i+1]) for i in range(len(rating_bounds)-1)]
    return np.matrix(one_vector)

def F(t,tenor,c,r):  #Forward value of the bond
    N = tenor.shape[0]
    #Eva
    Term1 =c*0.5*np.sum([[np.exp(-r(tenor[i][j]-t))*(tenor[i][j]>=t) for j in range(len(tenor[i]))] for i in range(N)],1)
    Term2 = np.exp([-r(tenor[i][-1]-t) for i in range(N)])
    #Term1 =c*0.5*np.sum(np.exp(-r(t-tenor[:][int(t*2):])))
    #Term2 = np.exp(-r(tenor[:][-1]-t))
    return Term1 +Term2 

def lambda_vector(ELGD,A,rho,c,g,D,s,trans_dict,T = 1
                  ,r =r_nelson,default_only=False,LGD_constant = True):
    # Reduce the vectors considered to a smaller dimension
    greater_zero = A>0
    A = A[greater_zero]
    ELGD = ELGD[greater_zero]
    rho = rho[greater_zero]
    c = c[greater_zero]
    g = g[greater_zero].astype(int)
    D = D[greater_zero]
    tenor = [np.array([i*0.5 for i in range((int(D[j]))*2+1)]) for j in range(len(A))]
    tenor = np.array(tenor,dtype = object)
    
    D = np.round(D*2,0)/2 # round to 0.5 digits
    #if s >0:
        #Eva:
    #Term1 = c*0.5*np.array([np.sum(np.exp(r (T - np.array(tenor[j][1:])[np.array(tenor[j][1:]) < T]))) for j in range(len(A))])
    Term1 = c*0.5*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T]) for j in range(len(A))])
    Term2 = F(T,tenor,c,r)
    Term3 = - np.array([np.sum([np.exp(-r(tenor[j][i]-T))*(ps_star(s[1:],0,tenor[j][i]-T,rho[j],trans_dict,default_only=default_only) \
                    -ps_star(s[1:],0,tenor[j][i-1]-T,rho[j],trans_dict,default_only=default_only))*(ELGD[j])*F(tenor[j][i],tenor,c,r)[j] for i in range(1,len(tenor[j])) if tenor[j][i]>T],0) for j in range(len(A))])
        #j = 1
    #i = 0
    #print("Array", [np.exp(-r(tenor[j][i]-T))*(ps_star(s[1:],0,tenor[j][i]-T,rho[j],trans_dict,default_only=default_only) \
    #                -ps_star(s[1:],0,tenor[j][i-1]-T,rho[j],trans_dict,default_only=default_only))*(ELGD[j])*F(tenor[j][i],tenor,c,r)[j] for i in range(1,len(tenor[j])) if tenor[j][i]>T])
    #print("term3", Term3)
    #Term3 = - np.reshape([np.sum([np.exp(-r(tenor[j][i]-T))*(ps_star(s,0,tenor[j][i]-T,rho[j]) \
    #                -ps_star(s,0,tenor[j][i-1]-T,rho[j]))*(ELGD[j])*F(tenor[j][i],tenor,c,r)[j] for i in range(1,len(tenor[j])) if tenor[j][i]>T],0) for j in range(len(A))],
    #                    (S+1,len(A)))
    #print([Term1,Term2,Term3])
        #Gordy
    #Term1 = c*0.5*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <=T])
    #Term2 = F(T)
    #Term3 = np.sum([np.exp(-r*(tenor[i]-T))*(ps_star(s,0,tenor[i]-T) \
    #            -ps_star(s,0,tenor[i-1]-T))*(F(tenor[i])-(1-ELGD)*(1+c/2)) for i in range(1,len(tenor)) if tenor[i]>T],0)
    if np.sum(Term3) == 0:
        Term3 = np.zeros((len(A),S))
    
    lam = (1/ P_0(ELGD,A,rho,c,g,D,tenor,trans_dict)[:,np.newaxis] )*(Term1[:,np.newaxis] + Term2[:,np.newaxis] +Term3)           
    

    #elif s==0:
        #Eva:
    #Expected Return in default state
    lam_0 = (1/ P_0(ELGD,A,rho,c,g,D,tenor,trans_dict))*((c/2)*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T])for j in range(len(A))])+(1-ELGD)*F(T,tenor,c,r))
    #LGD constant
    #lam_0 = (1-ELGD)
    #print(lam_0.shape)
        #Gordy:
        #lam = (1/ P_0())*((c/2)*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <T])+(1-ELGD)*(1+c/2))
    return np.transpose(np.concatenate([lam_0[:,np.newaxis],lam],axis =1))

def lambda_vector_terms_const_1(ELGD,A,rho,c,g,D,s,trans_dict,T = 1
                  ,r =r_nelson,default_only=False):
    # Reduce the vectors considered to a smaller dimension
    greater_zero = A>0
    A = A[greater_zero]
    rho = rho[greater_zero]
    c = c[greater_zero]
    g = g[greater_zero].astype(int)
    D = D[greater_zero]
    tenor = [np.array([i*0.5 for i in range((int(D[j]))*2+1)]) for j in range(len(A))]
    tenor = np.array(tenor,dtype = object)
    
    D = np.round(D*2,0)/2 # round to 0.5 digits
    #if s >0:
        #Eva:
    #Term1 = c*0.5*np.array([np.sum(np.exp(r (T - np.array(tenor[j][1:])[np.array(tenor[j][1:]) < T]))) for j in range(len(A))])
    Term1 = c*0.5*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T]) for j in range(len(A))])
    Term2 = F(T,tenor,c,r)
    Term3 = - np.array([np.sum([np.exp(-r(tenor[j][i]-T))*(ps_star(s[1:],0,tenor[j][i]-T,rho[j],trans_dict,default_only=default_only) \
                    -ps_star(s[1:],0,tenor[j][i-1]-T,rho[j],trans_dict,default_only=default_only))*F(tenor[j][i],tenor,c,r)[j] for i in range(1,len(tenor[j])) if tenor[j][i]>T],0) for j in range(len(A))])
    
    P_00 = P_0(ELGD,A,rho,c,g,D,tenor,trans_dict)
    lam_01 = (1/ P_00)*((c/2)*np.array([np.sum([np.exp(r(T-t_i)) for t_i in tenor[j][1:] if t_i <T])for j in range(len(A))]))
    lam_02 = F(T,tenor,c,r)
    
    return Term1,Term2,Term3,tenor,lam_01,lam_02,P_00

def lambda_vector_terms_const_2(ELGD,A,rho,c,g,D,s,trans_dict,Term1,Term2,Term3,tenor,lam_01,lam_02,lam_03,P_00,T = 1
                  ,r =r_nelson):
    lam_0 = lam_01+lam_02-lam_03
    if np.sum(Term3) == 0:
         Term3 = np.zeros((len(A),S))
     
    lam = (1/ P_00[:,np.newaxis] )*(Term1[:,np.newaxis] + Term2[:,np.newaxis] +Term3)           
     
    
     #elif s==0:
         #Eva:
     #Expected Return in default state
        #LGD constant
     #lam_0 = (1-ELGD)
     #print(lam_0.shape)
         #Gordy:
         #lam = (1/ P_0())*((c/2)*np.sum([np.exp(r*(T-t_i)) for t_i in tenor[1:] if t_i <T])+(1-ELGD)*(1+c/2))
    return np.transpose(np.concatenate([lam_0[:,np.newaxis],lam],axis =1))
    


def P_0(ELGD,A,rho,c,g,D,tenor,trans_dict,r =r_nelson, default_only=False):# Input are Vectors, s is a single state
    Term1 = F(0,tenor,c,r)
    # Eva:
    Term2 = -np.array([np.sum([np.exp(-r(tenor[j][i]))*(ps_star(g,0,tenor[j][i],rho,trans_dict,default_only=default_only)[j] \
                    -ps_star(g,0,tenor[j][i-1],rho,trans_dict,default_only=default_only)[j])*(F(tenor[j][i],tenor,c,r)[j]*ELGD[j]) for i in range(1,len(tenor[j]))]) for j in range(len(A))])
    # Gordy:
    #Term2 = -np.sum([np.exp(-r*(tenor[i]))*(ps_star(g,0,tenor[i]) \
    #                -ps_star(g,0,tenor[i-1]))*(F(tenor[i])-(1-ELGD)*(1+c/2)) for i in range(1,len(tenor))],0)
    val = Term1 + Term2
    return val #Bond price for each obligor (N-dim vector)

def rating_bounds(ELGD,A,rho,c,g,D,trans_dict,default_only=False):
    return [C(ELGD,A,rho,c,g,D,s,trans_dict,default_only=default_only) for s in range(-1,S+1)]  

def pi_func(X,rho,bounds):
    all_states = norm.cdf((bounds-X*np.sqrt(rho))/(np.sqrt(1-rho)))
    return np.diff(all_states,axis = 0) # (S+1,100)

def Z(ELGD,A,rho,c,g,Y,rating_bounds_computed):# Input are Vectors, s is a single state
    one_vector = [(Y>=rating_bounds_computed[i])*(Y<=rating_bounds_computed[i+1]) for i in range(len(rating_bounds_computed)-1)]
    return np.matrix(one_vector) # (S+1,100)

def Z_S(ELGD,A,rho,c,g,X,bounds,t,lambda_0):# Input are Vectors, s is a single state
    probs = np.exp((-t)*np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0))*pi_func(X,rho,bounds) 
    probs = probs/np.repeat(np.expand_dims(np.sum(probs,axis =0),0),S+1,axis = 0)
    #states = np.array([np.random.choice(a = np.arange(0,S+1),size = 1, p = probs[:,i])[0] for i in range(len(A))])
    
    # Convert states to one-hot matrix:
    rr = np.expand_dims(np.random.rand(probs.shape[1]), axis=0)
    states =  (probs.cumsum(axis=0) > rr).argmax(axis=0)        
    matt = np.zeros([S+1,len(A)])
    matt[states,np.arange(0,len(A))]=1
    return np.matrix(matt)

def to_optimize_mu(y,rho,lambda_0,ELGD,A,c,g,bounds):
    # Constant Approximation proposed in Glassermann and Li 2005, Section 5.1
    Constraints = [{'type': 'ineq', 'fun': lambda x: -np.sum(np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0)*pi_func(x,rho,bounds))-y}]
    mu_opt = minimize(lambda x: x**2,
                                   -1,
                                   constraints = Constraints)
    #print(mu_opt)
    return mu_opt.x

def M_L(ELGD,A,rho,c,g,X,t,lambda_0,bounds,threshold = 1e10):
    array_1 = np.exp(-t*np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0))*pi_func(X,rho,bounds)
    return np.prod(np.sum(array_1,axis = 0)) # Scalar
    #return np.prod(np.prod(array_1,axis = 0))

def r_mu(X,mu):
    val = np.exp(-mu*X+0.5*mu**2)
    return val

def t(ELGD,A,rho,c,g,D,X,y,bounds,lambda_0,bound_low = -500,bound_high =500):
    def to_solve(t):
        numerator_1 =  np.exp(-t*np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0))*pi_func(X,rho,bounds)
        numerator = np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0)*numerator_1
        denominator_1 = np.sum(np.exp(-t*np.repeat(np.expand_dims(A, 0),S+1,axis = 0)*np.array(lambda_0))*pi_func(X,rho,bounds),axis = 0) # 100
        denominator = np.repeat(np.expand_dims(denominator_1, 0),S+1,axis = 0)   
        return np.abs(y-np.sum(-numerator/denominator))
    solution = minimize(to_solve,0.1,bounds  = [(bound_low,bound_high)])
    #solution =    fsolve(to_solve,100,full_output=0)
    #print(fsolve(to_solve,100,full_output=True))

    return solution.x

