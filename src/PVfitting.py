import numpy as np
from scipy.constants import Boltzmann as k, eV as q
from sympy import Eq, linear_eq_to_matrix, simplify, expand, exp, LambertW as W
from PVgeneral import PVgeneral
eps = np.finfo(float).eps

class PVfitting(PVgeneral):
  def __init__(self, gamma=1.0):
    
    super().__init__(gamma)

    # determination of the dependence of IL, I0 and Gp on Rs and b
    [self.IL_LVK, self.I0_LVK, self.Gp_LVK] = self.get_dependence_IL_I0_Gp
      
    # construction of the temperature system
    print("Generation of the system to obtain the thermal parameters")
    [self.A_temp, self.b_temp] = self.get_temperature_system(model='11PFF')

    # diff dP/dV = 0
    self.F0 = self.get_F0



  def loss(self, z, only_cost=True):
    rho0 = np.log1p(z) #  => log(z+1) evita el cero
    if only_cost:
      return rho0
    else:
      t = 1 + z
      rho1 = 1 / t
      rho2 = -1 / t**2
      rho = np.stack([rho0, rho1, rho2], axis=0)
      return rho

  
  def get_temperature_system(self, model='11PFF'):
    # unpacking variables
    alphaT, deltaI0, deltaRs = self.T_coefficients
    alpha_spec, beta_spec, gamma_spec = self.spec
    
    # linear system construction
    # In order to construct the linear system, it is necessary 
    # that the Eg component of the derivative of I0 is zero. This is 
    # achieved if and only if one works at the reference point.  
    dIpv_oc = self.get_dIpv_oc(model).subs({self.T:self.Tref, self.S:self.Sref})
    dIpv_sc = self.get_dIpv_sc(model).subs({self.T:self.Tref, self.S:self.Sref})
    dIpv_mp = self.get_dIpv_mp(model).subs({self.T:self.Tref, self.S:self.Sref})
    dVpv_mp = self.get_dVpv_mp(model).subs({self.T:self.Tref, self.S:self.Sref})
    dPpv_mp = self.get_dPpv_mp(model).subs({self.T:self.Tref, self.S:self.Sref})

    Ax, bx = linear_eq_to_matrix([
                Eq(dIpv_oc, 0), 
                Eq(dIpv_sc, 0), 
                Eq(dIpv_mp, 0), 
                Eq(dVpv_mp, 0),
                Eq(dPpv_mp, gamma_spec)], [alphaT, deltaI0, deltaRs, self.gammaImp, self.gammaVmp])
    Ax = simplify(expand(Ax))
    bx = simplify(expand(bx))
    return Ax, bx

  def compute_temperature_coefficients(self, params_ref):
    """ 
    Determination of temperature coefficients:
     -> is only applicable at reference point (1), 
        because at this point the system is linear
    """
    # unpacking variables
    [alpha_spec, beta_spec, gamma_spec] = self.spec_Tcoeff

    # replace values
    # at the reference point, reference parameters are equal to operation value (x=x_ref)
    
    A_T = self.A_T1.subs(dict(zip(self.params_ref, params_ref)))
    b_T = self.b_T1.subs(dict(zip(self.params_ref, params_ref)))
    b_T = b_T.subs(dict(zip(self.spec, [alpha_spec, beta_spec, gamma_spec])))
    try:
      T_solve = A_T.solve(b_T)
    except Exception as e:
      print(e)
      T_solve = A_T.solve(b_T)
    alphaT, deltaI0, deltaRs, gammaImp, gammaVmp = [float(val) for val in T_solve]
    return [alphaT, deltaI0, deltaRs, gammaImp, gammaVmp]

    
  def b_limits(self, T_ref, n_max, n_min, Ns):    
    # definition of b1_lb and b1_ub
    bfun = q/self.n/Ns/k/T_ref

    # compute bref limits 
    b_min = float(bfun.subs({self.n:n_max}))
    b_max = float(bfun.subs({self.n:n_min}))
    return [b_min, b_max]

    
  def Rs_limits(self, bref, T_ref, T, zip_meas2ref0):
    # unpacking variables
    b_ref = self.params_ref[0]
    b     = self.params[0]
    Voc, Isc, Vmp, Imp = self.meas

    # definition of Rs_max
    V0 = 1/b
    u = W( -exp( (Voc-2*Vmp-V0)/V0 ), -1)
    # u = W( -exp( (Voc-2*Vmp-V0)/V0 ))
    Rs_max_fun = V0/Imp * (1+ u) + Vmp/Imp

    # compute b
    b_ref0 = self.modelSym11PFF.bfun(T_ref).subs({self.Tref:T_ref, b_ref:bref})
    b_refx = self.modelSym11PFF.bfun(T)    .subs({self.Tref:T_ref, b_ref:bref})
    

    # compute Rs_max
    Rs_ref0_ub = np.real(Rs_max_fun.subs(self.zip_meas_to_ds1).subs({b:b_ref0}))
    Rs_refx_ub = np.real(Rs_max_fun.subs(zip_meas2ref0).subs({b:b_refx}))
    
      
    # compute Rs_min
    Rs_ref0_lb = self.Rs_min*(Rs_refx_ub/Rs_ref0_ub)
    return float(Rs_ref0_lb), float(Rs_ref0_ub)


  @property
  def k11PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'mI0', 'mRs', 'mGp', 
            'alphaT', 'deltaI0', 'deltaRs', 
            'T_ref', 'S_ref']
  
  @property
  def k7PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT',# alphaIsc
            'mI0', 'deltaRs', 
            'T_ref', 'S_ref']
  
  @property
  def k6PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT', # alphaIsc
            'mIL',
            'T_ref', 'S_ref']
  
  @property
  def k5PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT',# alphaIsc
            'T_ref', 'S_ref']











































