from scipy.constants import Boltzmann as k, eV as q, zero_Celsius as T0
from sympy import exp, Pow, Symbol
import numpy as np
import mpmath, decimal
eps = np.finfo(float).eps
mpmath.mp.dps = 100
decimal.getcontext().prec = 20
eps = np.finfo(float).eps

def lambertw(x):
  return np.real(np.longdouble(mpmath.lambertw(x)))

class PVModel: 
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref):
    self.b_ref = b_ref
    self.IL_ref = IL_ref
    self.I0_ref = I0_ref
    self.Rs_ref = Rs_ref
    self.Gp_ref = Gp_ref
    self.T_ref = T_ref 
    self.S_ref = S_ref
    self.EgV   = 1.121
    
  def __call__(self, S, T, pts=100, V_model=None):
    [Isc, Vsc, Imp, Vmp, Pmp, Ioc, Voc] = self.predict(S, T)
    if V_model is None:
      V_model = np.linspace(0, Voc, pts)
    else:
      if Voc>V_model[-1]:
        nsamples = int((Voc-V_model[-1])//0.5)
        dVoc = np.linspace(V_model[-1], Voc, nsamples)
        V_model = np.concatenate([V_model, dVoc])
    I_model = []
    for Vm in V_model:
      I_model.append(self.fun_Ipv(Vm))
    I_model = np.array(I_model).reshape(V_model.shape)
    P_model = V_model*I_model

    if np.isnan(np.array([Isc, Vsc, Imp, Vmp, Pmp, Ioc, Voc])).any():
      # rescues the values of the curve
      Isc = I_model[0]
      Vsc = V_model[0]

      idx_max = P_model.argmax()
      Imp = I_model[idx_max]
      Vmp = V_model[idx_max]
      Pmp = P_model[idx_max]
  
      Ioc = I_model[-1]
      Voc = V_model[-1]

    else:
    
        
      try:
        if not np.abs(1 - I_model[P_model.argmax()]/Imp)<0.05: 
          # rescues the values of the curve
          Isc = I_model[0]
          Vsc = V_model[0]

          Vmp = V_model[P_model.argmax()]
          Imp = I_model[P_model.argmax()]
          Pmp = P_model[P_model.argmax()]

          Ioc = I_model[-1]
          Voc = V_model[-1]
        
      except Exception as e:
          pass

    
    return [[Isc, Vsc, Imp, Vmp, Pmp, Ioc, Voc], [V_model, I_model, P_model]]
  
  def update_params(self, S, T):
    self.b, self.IL, self.I0, self.Rs, self.Gp = self.params(S,T)

  def Eg(self, T):
    return self.EgV*(1-0.0002677*(T-self.T_ref))  
  
  def auxiliar(self, T):
    return q/k *(self.Eg(self.T_ref)/self.T_ref-self.Eg(T)/T)  
        
  def fun_dPdV(self, Vmp):
    """
    dP/dV = I + V dI/dV <=> I + V *I'(V)

    where
    => (1) in MPP, dP/dV=0
    => (2) Ipv = (IL+I0-Gp*Vpv)/(Rs*Gp+1) - W0/(b*Rs)
    => (3)  W0 = W( b*Rs*I0/(Rs*Gp+1)* exp( b*(Rs*(IL+I0)+Vpv)/(Rs*Gp+1) ) )
       where
        => K1 = Rs*I0
        => K2 = Rs*(IL+I0)
        => K3 = b/(Rs*Gp+1)
           W0 = W( K1*K3 exp( K3*(K2+V) ) )
       dWx/dx = Wx/(x*(Wx+1)) 

    Apply (1) and (2):
      => Imp = - Vmp I'(Vmp)
    Thus
      => dI/dV = -Gp/(Rs*Gp+1) - 1/(b*Rs) dW0/dV
      where
        if f1(V) =    K1*K3 exp( K3*(K2+V) )
          df1/dV = K1*K3**2 exp( K3*(K2+V) )
                 = K3* f1(V)

          dW0/dV = W(f1)/(f1*(W(f1)+1)) df1/dV
                 = W0/(f1*(W0+1)) * K3*f1
                 = W0*K3/(W0+1)
                 = W0/(W0+1)*b/(Rs*Gp+1)


      => dI/dV = -Gp/(Rs*Gp+1) - 1/(b*Rs) W0/(W0+1)*b/(Rs*Gp+1)
               = -Gp/(Rs*Gp+1) - W0/(W0+1)/(Rs*Gp+1)/Rs
               = -Gp/(Rs*Gp+1) * (1 + W0/(W0+1)/Rs)
    
    => Imp = - Vmp I'(Vmp)
           = - Vmp/(Rs*Gp+1) * (Gp + W0/(W0+1)/Rs)
    """
    return Vmp/(self.Rs*self.Gp + 1)*(self.Gp+self.W0(Vmp)/(1+self.W0(Vmp))/self.Rs    ) 
  
  def fun_Vpv(self, Ipv):
    """
    Vpv = (IL+I0-(1+Gp*Rs)*Ipv)/Gp-lambertw(b*I0/Gp*exp(b*(IL+I0-Ipv)/Gp))/b
    =============================
    function decomposition
    =============================
    K1 = (IL+I0-(1+Gp*Rs)*Ipv)/Gp
    K2 = b*I0/Gp
    K3 = b*(IL+I0-Ipv)/Gp
    K4 = K2*exp(K3)
    if apply log in K4:
        log(K4) = log(K2*exp(K3))
        log(K4) = log(K2) + log(exp(K3))
        log(K4) = log(K2) + K3
             K4 = exp(log(K2) + K3)
    Vpv = K1 - lambertw(K4)/b
    """
    K1 = (self.IL+self.I0-(1+self.Gp*self.Rs)*Ipv)/self.Gp
    K2 = self.b*self.I0/self.Gp
    K3 = self.b*(self.IL+self.I0-Ipv)/self.Gp
    K4 = K2*mpmath.exp(K3)
    return K1 - lambertw(K4)/self.b

  def fun_Vpv2(self, Ipv, step_tol=1e-8):
    """
    ======================================================================================
    Alternative method for estimating Voc: iterative lambertw
    ======================================================================================
    Vpv = (IL+I0-(1+Gp*Rs)*Ipv)/Gp-lambertw(b*I0/Gp*exp(b*(IL+I0-Ipv)/Gp))/b
    
    => function decomposition
    K1 = (IL+I0-(1+Gp*Rs)*Ipv)/Gp
    K2 = b*I0/Gp
    K3 = b*(IL+I0-Ipv)/Gp
    K4 = K2*exp(K3)
    """
    def iterative_lambertw(z):
      # auxiliary value for the lambert functionw
      w = mpmath.ln(z+1)
      step = w

      # Estimation of the value associated with the lambert functionw
      while mpmath.absmax(step) > step_tol:
        ew = mpmath.exp(w)
        numer = (w*ew - z)
        step = numer/(ew*(w+1) - (w+2)*numer/(2*w + 2))
        w -=  step
      return w

    K1 = (self.IL+self.I0-(1+self.Gp*self.Rs)*Ipv)/self.Gp
    K2 = self.b*self.I0/self.Gp
    K3 = self.b*(self.IL+self.I0-Ipv)/self.Gp
    if isinstance(K2, np.longdouble):
      z = K2*mpmath.exp(K3)
      w = iterative_lambertw(z)
      W_val = np.asarray(w).reshape([-1,1])
    else:
      W_val = []
      for n in range(len(K2)):
        z = K2[n][0]*mpmath.exp(K3[n][0])
        w = iterative_lambertw(z)
        W_val.append(w)
      W_val = np.asarray(W_val).reshape([-1,1])
    
    # Estimated Voc
    return K1-(W_val/self.b).astype(np.longdouble)
  
  def fun_Ipv(self, Vpv):
    """
    Ipv = (IL+I0-Gp*Vpv)/(Rs*Gp+1)-W0(Vpv)/(b*Rs)
    =============================
    function decomposition
    =============================
    K1 = (IL+I0-Gp*Vpv)
    K2 = Rs*Gp+1
    K3 = b*Rs
    """
    K1 = self.IL+self.I0-self.Gp*Vpv
    K2 = self.Rs*self.Gp+1
    K3 = self.b*self.Rs
    return K1/K2 - self.W0(Vpv)/K3
  
  def W0(self, Vpv):
    """
    W0 = W(b*Rs*I0/(Rs*Gp+1)*exp(b*(Rs*(IL+I0)+Vpv)/(Rs*Gp+1)))
    =============================
    function decomposition
    =============================
    K1 = b*Rs*I0
    K2 = Rs*Gp+1
    K3 = b*(Rs*(IL+I0)+Vpv)
    K4 = K1/K2*exp(K3/K2)
    if apply log in K4:
          log(K4) = log(K1/K2*exp(K3/K2))
          log(K4) = log(K1/K2) + log(exp(K3/K2))
          log(K4) = log(K1/K2) + K3/K2
      =>  K4 = exp(log(K1/K2) + K3/K2)
    W0 = lambertw(K4)
    """
    K1 = self.b*self.Rs*self.I0
    K2 = self.Rs*self.Gp+1
    K3 = self.b*(self.Rs*(self.IL+self.I0)+Vpv)
    K4 = (K1/K2)*mpmath.exp(K3/K2)
    return lambertw(K4)
  
    



  def predict(self, S, T, MaxIterations=5000, tol=1e-9, alpha=0.15, beta=0.85, VmpIni=None): 
    self.update_params(S, T)
    Isc = self.fun_Ipv(0.)
    Vsc = self.fun_Vpv(Isc)
    Voc = self.fun_Vpv(0.)
    if np.isnan(Voc):
      Voc = float(self.fun_Vpv2(0.).flatten()[0])
    Ioc = self.fun_Ipv(Voc)   
    adjust_Vmp = lambda Vmp: self.fun_Ipv(Vmp)-self.fun_dPdV(Vmp)
    IniIterations = 0
    if VmpIni==None: 
      Vmp0, Vmp1 = Voc*alpha, Voc*beta
    else: 
      Vmp0, Vmp1 = VmpIni
    error = np.abs(Vmp1-Vmp0)
    while error>np.float64(tol):
      IniIterations+=1
      diff_Vmp = Vmp1-Vmp0
      diff_FVmp = adjust_Vmp(Vmp1)-adjust_Vmp(Vmp0)
      Vmp11 = Vmp0 - adjust_Vmp(Vmp0)*diff_Vmp/diff_FVmp
      Vmp0, Vmp1 = Vmp1, Vmp11
      error = np.abs(Vmp1-Vmp0)
      if IniIterations==MaxIterations:
        break
    Vmp = Vmp1
    Imp = self.fun_Ipv(Vmp)
    Pmp = Vmp*Imp
    return [float(meas) for meas in [Isc, Vsc, Imp, Vmp, Pmp, Ioc, Voc]]
  









  
class Model5PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref,Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
      
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(T),
      self.Rsfun(S),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)*(self.IL_ref+self.alphaT*(T-self.T_ref))
  
  def I0fun(self, T):
    return self.I0_ref*(T/self.T_ref)**3*np.exp(self.auxiliar(T))

  def Rsfun(self, S):
    return self.Rs_ref*(S/S) # to vector

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
  
class ModelA5PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, Adjust, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
    self.Adjust = Adjust

  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(T),
      self.Rsfun(S),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)*(self.IL_ref+self.alphaT*(1-self.Adjust/100)*(T-self.T_ref))
  
  def I0fun(self, T):
    return self.I0_ref*(T/self.T_ref)**3*np.exp(self.auxiliar(T))

  def Rsfun(self, S):
    return self.Rs_ref*(S/S) # to vector

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
  
class Model6PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, mIL, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
    self.mIL = mIL
      
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(T),
      self.Rsfun(S),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    return self.b_ref*(self.T_ref/T)
  
  def ILfun(self, S, T):
    return (S/self.S_ref)**self.mIL*(self.IL_ref+self.alphaT*(T-self.T_ref))
  
  def I0fun(self, T):
    return self.I0_ref*(T/self.T_ref)**3*np.exp(self.auxiliar(T))

  def Rsfun(self, S):
    return self.Rs_ref*(S/S) # to vector

  def Gpfun(self, S):
    return self.Gp_ref*(S/self.S_ref)
     
class Model7PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref, alphaT, mI0, deltaRs, T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.alphaT = alphaT
    self.mI0 = mI0
    self.deltaRs = deltaRs
      
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(S, T),
      self.Rsfun(T),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    """
    ======================================================================================
      b(T) = b_ref*(T_ref/T)
    ======================================================================================
    """
    if isinstance(self.b_ref, Symbol):
      T_long = Pow(self.T_ref/T, 1)
      return self.b_ref*T_long
    else:
      T_long = mpmath.power(self.T_ref/T, 1)
      return np.longdouble(self.b_ref*T_long)
  
  def ILfun(self, S, T):
    """
    ======================================================================================
      IL(S, T) = (S/S_ref)*(IL_ref+alphaT*(T-T_ref))
    ======================================================================================
    """
    if isinstance(self.IL_ref, Symbol):
      S_long = Pow(S/self.S_ref, 1)
      T_long = self.alphaT*(T-self.T_ref)
      return S_long*(self.IL_ref+T_long)
    else:
      S_long = mpmath.power(S/self.S_ref, 1)
      T_long = self.alphaT*(T-self.T_ref)
      return np.longdouble(S_long*(self.IL_ref+T_long))

  def I0fun(self, S, T):
    """
    ======================================================================================
      I0(S, T) = I0_ref*(S/S_ref)**mI0*(T/T_ref)**3*exp(auxiliar(T))
    ======================================================================================
    """
    if isinstance(self.I0_ref, Symbol):
      S_long  = Pow(S/self.S_ref, self.mI0)
      T_long  = Pow(T/self.T_ref, 3)
      T_long *= exp(self.auxiliar(T))
      return self.I0_ref*S_long*T_long
    else:
      S_long  = mpmath.power(S/self.S_ref, self.mI0)
      T_long  = mpmath.power(T/self.T_ref, 3)
      T_long *= mpmath.exp(self.auxiliar(T))
      return np.longdouble(self.I0_ref*S_long*T_long)

  def Rsfun(self, T):
    """
    ======================================================================================
      Rs(S, T) = Rs_ref*exp(deltaRs*(T-T_ref))
    ======================================================================================
    """
    if isinstance(self.Rs_ref, Symbol):
      T_long = exp((T-self.T_ref)*self.deltaRs)
      return self.Rs_ref*T_long
    else:
      T_long = mpmath.exp((T-self.T_ref)*self.deltaRs)
      return np.longdouble(self.Rs_ref*T_long)

  def Gpfun(self, S):
    """
    ======================================================================================
      Gp(S) = Gp_ref*(S/S_ref)
    ======================================================================================
    """
    if isinstance(self.Gp_ref, Symbol):
      S_long = Pow(S/self.S_ref, 1)
      return self.Gp_ref*S_long
    else:
      S_long = mpmath.power(S/self.S_ref, 1)
      return np.longdouble(self.Gp_ref*S_long)

class Model11PFF(PVModel):
  def __init__(self, b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref,
                     mI0, mRs, mGp,
                     alphaT, deltaI0, deltaRs,
                     T_ref, S_ref):
    super().__init__(b_ref,IL_ref, I0_ref, Rs_ref, Gp_ref, T_ref, S_ref)
    self.mI0 = mI0
    self.mRs = mRs
    self.mGp = mGp
    self.alphaT = alphaT
    self.deltaI0 = deltaI0
    self.deltaRs = deltaRs
    
  def params(self, S, T):
    return [
      self.bfun(T),
      self.ILfun(S, T),
      self.I0fun(S, T),
      self.Rsfun(S, T),
      self.Gpfun(S),
    ]

  def bfun(self, T):
    """
    ======================================================================================
      b(T) = b_ref*(T_ref/T)
    ======================================================================================
    """
    if isinstance(self.b_ref, Symbol):
      T_long = Pow(self.T_ref/T, 1)
      return self.b_ref*T_long
    else:
      T_long = mpmath.power(self.T_ref/T, 1)
      return np.longdouble(self.b_ref*T_long)
  
  def ILfun(self, S, T):
    """
    ======================================================================================
      IL(S, T) = (S/S_ref)*(IL_ref+alphaT*(T-T_ref))
    ======================================================================================
    """
    if isinstance(self.IL_ref, Symbol):
      S_long = Pow(S/self.S_ref, 1)
      T_long = self.alphaT*(T-self.T_ref)
      return S_long*(self.IL_ref+T_long)
    
    else:
      S_long = mpmath.power(S/self.S_ref, 1)
      T_long = self.alphaT*(T-self.T_ref)
      return np.longdouble(S_long*(self.IL_ref+T_long))
        
  def I0fun(self, S, T):
    """
    ======================================================================================
      I0(S, T) = I0_ref*(S/S_ref)**mI0*(T/T_ref)**3*exp(deltaI0*auxiliar(T))
    ======================================================================================
    """
    if isinstance(self.I0_ref, Symbol):
      S_long  = Pow(S/self.S_ref, self.mI0)
      T_long  = Pow(T/self.T_ref, 3)
      T_long *= exp(self.deltaI0*self.auxiliar(T))
      return self.I0_ref*S_long*T_long
    else:
      S_long  = mpmath.power(S/self.S_ref, self.mI0)
      T_long  = mpmath.power(T/self.T_ref, 3)
      T_long *= mpmath.exp(self.deltaI0*self.auxiliar(T))
      return np.longdouble(self.I0_ref*S_long*T_long)

  def Rsfun(self, S, T):
    """
    ======================================================================================
      Rs(S, T) = Rs_ref*(S/S_ref)**mRs*exp(deltaRs*(T-T_ref))
    ======================================================================================
    """
    if isinstance(self.Rs_ref, Symbol):
      S_long = Pow(S/self.S_ref, self.mRs)
      T_long = exp((T-self.T_ref)*self.deltaRs)
      return self.Rs_ref*S_long*T_long
    else:
      S_long = mpmath.power(S/self.S_ref, self.mRs)
      T_long = mpmath.exp((T-self.T_ref)*self.deltaRs)
      return np.longdouble(self.Rs_ref*S_long*T_long)

  def Gpfun(self, S):
    """
    ======================================================================================
      Gp(S) = Gp_ref*(S/S_ref)**mGp
    ======================================================================================
    """
    if isinstance(self.Gp_ref, Symbol):
      S_long = Pow(S/self.S_ref, self.mGp)
      return self.Gp_ref*S_long
    else:
      S_long = mpmath.power(S/self.S_ref, self.mGp)
      return np.longdouble(self.Gp_ref*S_long)