import pandas as pd 
import numpy as np
from time import perf_counter
from IPython.utils import io
import swifter 
from least_squares_dynamics_bounds.least_squares import least_squares
from scipy.constants import Boltzmann as k, eV as q
from sympy import symbols, lambdify, solve, Eq, log
from PVfitting import PVfitting


eps = np.finfo(float).eps

class PVfitting_11PFF_2P(PVfitting):  
  S1, T1 = symbols("S_{1}, T_{1}", real=True)
  S2, T2 = symbols("S_{2}, T_{2}", real=True)
  S3, T3 = symbols("S_{3}, T_{3}", real=True)
  
  meas1 = symbols("V_{oc-1}, I_{sc-1}, V_{mp-1}, I_{mp-1}", positive=True, real=True)
  meas2 = symbols("V_{oc-2}, I_{sc-2}, V_{mp-2}, I_{mp-2}", positive=True, real=True)
  meas3 = symbols("V_{oc-3}, I_{sc-3}, V_{mp-3}, I_{mp-3}", positive=True, real=True)
  
  params1 = symbols("b_{1}, I_{L-1}, I_{0-1}, R_{s-1}, G_{p-1}", positive=True, real=True)
  params2 = symbols("b_{2}, I_{L-2}, I_{0-2}, R_{s-2}, G_{p-2}", positive=True, real=True)
  params3 = symbols("b_{3}, I_{L-3}, I_{0-3}, R_{s-3}, G_{p-3}", positive=True, real=True)

  
  def zip_meas_2_dsx(self, meas):
    return dict(zip(self.meas, meas))
    
  @property    
  def zip_meas_to_ds1(self):  
    return self.zip_meas_2_dsx(self.meas_ds1[2:-1])

  @property
  def zip_meas_to_ds2(self):
    return self.zip_meas_2_dsx(self.meas_ds2[2:-1])

  @property
  def zip_meas_to_ds3(self):
    return self.zip_meas_2_dsx(self.meas_ds3[2:-1])

  @property    
  def zip_meas1_to_ds1(self):  
    return dict(zip([self.S1]+ [self.T1] + list(self.meas1), self.meas_ds1[:-1]))

  @property    
  def zip_meas2_to_ds2(self):  
    return dict(zip([self.S2]+ [self.T2] + list(self.meas2), self.meas_ds2[:-1]))

  @property    
  def zip_meas3_to_ds3(self):  
    return dict(zip([self.S3]+ [self.T3] + list(self.meas3), self.meas_ds3[:-1]))





  def __call__(self, PVModule, pto1, pto2, pto3, 
               n_max=4.5, n_min=0.5,  
               regularizer=True, regularizer_mS=True, regularizer_mT=True,
               use_presolve=True, pts_presolve=20, 
               Rs_min = 1e-6,
               tol=1e-9, weigth=0.8):
    self.pto1 = pto1
    self.pto2 = pto2
    self.pto3 = pto3
    self.Rs_min = Rs_min
    Ns, Np = self.Ns_Np.loc[PVModule]  
    self.compile(PVModule, Ns, pto1, pto2, pto3, 
                 n_max=n_max, n_min=n_min, 
                 regularizer=regularizer)

    self.regularizer_mS = regularizer_mS
    self.regularizer_mT = regularizer_mT
    self.w = weigth
    self.mS_max = 5
    self.mT_max = 5

    if self.regularizer:
        if self.regularizer_mS and self.regularizer_mT:
            self.weigths = [self.w, self.w, self.w, 1-self.w, 1-self.w, 1-self.w, 1-self.w, 1-self.w]
        elif self.regularizer_mS or self.regularizer_mT:
            self.weigths = [self.w, self.w, self.w, 1-self.w, 1-self.w, 1-self.w, 1-self.w]
        else:
            self.weigths = [self.w, self.w, self.w, 1-self.w, 1-self.w, 1-self.w]
    else:
        self.weigths = [1, 1, 1]

      
    # get preliminary result by meshing
    time0 = perf_counter()
    print()
    print('get initial guess')
    df_grid = self.get_initial_guess(pts_presolve, use_presolve=use_presolve)

    fil0 = df_grid.ePlus == 0
    if self.regularizer:
        fil1 = df_grid.loc[fil0].loss_reg == df_grid.loc[fil0].loss_reg.min()
    else:
        fil1 = df_grid.loc[fil0].loss == df_grid.loc[fil0].loss.min()
    b1_guess, Rs1_guess, Rs2_guess, loss_guess, loss_reg_guess = df_grid.loc[fil0].loc[fil1, ['b1', 'Rs1', 'Rs2', 'loss', 'loss_reg']].values[0]

    try:
      params_guess, epsF = self.eval_eqs([b1_guess, Rs1_guess, Rs2_guess], verbose=True)
    except:
      pass
      
    # refinement of the result by solver
    print()
    print('least_squares')
    dfx, df = self.get_params([b1_guess, Rs1_guess, Rs2_guess], tol=tol)
    time1 = perf_counter()

    dfx.insert(0, 'PVModule', PVModule)
    dfx.insert(1, 'pto1', pto1)
    dfx.insert(2, 'pto2', pto2)
    dfx.insert(3, 'b_0', b1_guess)
    dfx.insert(4, 'Rs1_0', Rs1_guess)
    dfx.insert(5, 'Rs2_0', Rs2_guess)
    if self.regularizer:
        dfx.insert(6, 'loss0', loss_reg_guess)
    else:
        dfx.insert(6, 'loss0', loss_guess)
    dfx.insert(8, 'time', time1-time0)
    return df, dfx, df_grid   

  def compute_irradiance_coefficients(self, params1, params2, T_params):      
    # unpacking variables
    S1, T1 = self.meas_ds1[:2]
    S2, T2 = self.meas_ds2[:2]      

    b1, IL1, I01, Rs1, Gp1 = params1
    b2, IL2, I02, Rs2, Gp2 = params2

    alphaT, deltaI0, deltaRs, gammaImp, gammaVmp = T_params

    # compute auxiliar params
    sum_I0  = 3*log(T1/T2)
    sum_I0 += q*deltaI0/k *( self.Eg(T2, T1)/T2 -self.Eg(T1, T1)/T1)
    sum_Rs  = deltaRs *(T1 - T2)

    # values 
    mI0v = log(I02/I01)+sum_I0
    mRsv = log(Rs2/Rs1)+sum_Rs
    mGpv = log(Gp2/Gp1)

    # compute coefficients
    mI0 = float(mI0v/log(S2/S1))
    mRs = float(mRsv/log(S2/S1))
    mGp = float(mGpv/log(S2/S1))
    return [mI0, mRs, mGp]


  def bound_callbacks(self, bref, iterative=False):
      T1 = self.meas_ds1[1] # Tref = T1
      Rs1_lb, Rs1_ub = self.Rs_limits(bref, T1, T1, self.zip_meas_to_ds1)
      
      T2 = self.meas_ds2[1]
      Rs2_lb, Rs2_ub = self.Rs_limits(bref, T1, T2, self.zip_meas_to_ds2)
      
      if self.regularizer:
        T3 = self.meas_ds3[1]
        Rs3_lb, Rs3_ub = self.Rs_limits(bref, T1, T3, self.zip_meas_to_ds3)
          
        lb = [self.b1_min, Rs1_lb, Rs2_lb, Rs3_lb]
        ub = [self.b1_max, Rs1_ub, Rs2_ub, Rs3_ub]          
      else:
          lb = [self.b1_min, Rs1_lb, Rs2_lb]
          ub = [self.b1_max, Rs1_ub, Rs2_ub]
      
      if iterative:
          return lb[:3], ub[:3]
      else:
          return lb, ub
    

    
  def compile(self, PVModule, Ns, 
              pto1, pto2, pto3=None, 
              n_max=4.5, n_min=0.5, 
              verbose=True, regularizer=True):

    # =========================================================
    # Load measurement and T_coefficients from datasheet
    # =========================================================
    # temperatura coefficients
    self.spec_Tcoeff = self.get_temperature_coefficients(PVModule, self.Tcoeff)
      
    data = self.DAta.loc[PVModule]
    data = data.sort_values(data.columns[0], ascending=False) 
    data = data.reset_index(drop=True)

    assert pto1 in data.Type.tolist() and pto2 in data.Type.tolist()    
    # get first point (reference)
    self.meas_ds1 = self.get_point(data, pto1)
    S_ref, T_ref = self.meas_ds1[:2]
    S1, T1 = self.meas_ds1[:2]

    # get second point (complementary)
    self.meas_ds2 = self.get_point(data, pto2)
    S2, T2 = self.meas_ds2[:2]

    assert not(self.meas_ds1[0] == self.meas_ds2[0]), """
    Irradiance of the operating points cannot be equal: undefined S_coefficients → log(S2/S1)"""

    if verbose:
        self.print_point(pto1, self.meas_ds1, 'Reference')
        self.print_point(pto2, self.meas_ds2, 'Complementary')
        
    if regularizer:
      try:
        # get third point (regulatory)
        self.meas_ds3 = self.get_point(data, pto3)
        S3, T3 = self.meas_ds3[:2]
        if verbose:
            self.print_point(pto3, self.meas_ds3, 'Regularizer')
        self.reg_monitor = True
        self.regularizer = True
        
      except:
        print("the PV module does not support a regularized solution (it has only two operating points)")
        self.reg_monitor = False
        self.regularizer = False
    else:
      self.regularizer = False
      try:
        # get third point (regulatory)
        self.meas_ds3 = self.get_point(data, pto3)
        S3, T3 = self.meas_ds3[:2]
        if verbose:
            self.print_point(pto3, self.meas_ds3, 'Regularizer')
        self.reg_monitor = True
      except:
        self.reg_monitor = False
    
    # =========================================================
    # Defined global bounds
    # =========================================================
    print()
    print('global bounds')
    [self.b1_min, self.b1_max] = self.b_limits(T_ref, n_max, n_min, Ns)
    self.lb, self.ub = self.bound_callbacks(self.b1_min)
    print('lb: {0}'.format(self.lb))
    print('ub: {0}'.format(self.ub))    
      
    # =================================================
    # unpacking variables and dictionaries
    # =================================================
    b, IL, I0, Rs, Gp = self.params
    b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref = self.params_ref
    alphaT, deltaI0, deltaRs = self.T_coefficients
    mI0, mRs, mGp = self.S_coefficients

    # all keys
    keys_ref = {self.Sref:S_ref, self.Tref:T_ref}
    
    keys_m1 = self.zip_meas1_to_ds1
    keys_m1.update(keys_ref)
      
    keys_m2 = self.zip_meas2_to_ds2
    keys_m2.update(keys_ref)
    
    if self.reg_monitor:
        keys_m3 = self.zip_meas3_to_ds3
        keys_m3.update(keys_ref)
    
    # =================================================
    # symbolic temperature system 
    # =================================================
    self.A_T1 = self.A_temp.subs(self.zip_meas_to_ds1).subs(keys_ref)
    self.b_T1 = self.b_temp.subs(self.zip_meas_to_ds1).subs(keys_ref)
    
    
    # =================================================
    # 11pff functions
    # =================================================
    self.bfun = self.modelSym.bfun(self.T).subs(keys_ref)
    self.bfun = lambdify([b_ref, self.T], self.bfun)
    
    self.ILfun = self.modelSym.ILfun(self.S, self.T).subs(keys_ref)
    self.ILfun = lambdify([IL_ref, alphaT, self.S, self.T], self.ILfun)
    
    self.I0fun = self.modelSym.I0fun(self.S, self.T).subs(keys_ref)
    self.I0fun = lambdify([I0_ref, deltaI0, mI0, self.S, self.T], self.I0fun)
    
    self.Rsfun = self.modelSym.Rsfun(self.S, self.T).subs(keys_ref)
    self.Rsfun = lambdify([Rs_ref, deltaRs, mRs, self.S, self.T], self.Rsfun)
    
    self.Gpfun = self.modelSym.Gpfun(self.S).subs(keys_ref)
    self.Gpfun = lambdify([Gp_ref, mGp, self.S], self.Gpfun)
    
    
    # =================================================
    # symbolic operation functions
    # =================================================
    # evaluating symbolic expressions in reference point
    IL1_LVK = self.IL_LVK.subs(keys_ref).subs(self.zip_meas_to_ds1)
    I01_LVK = self.I0_LVK.subs(keys_ref).subs(self.zip_meas_to_ds1)
    Gp1_LVK = self.Gp_LVK.subs(keys_ref).subs(self.zip_meas_to_ds1)
    self.IL1_LVK = lambdify([b, Rs], IL1_LVK)
    self.I01_LVK = lambdify([b, Rs], I01_LVK)
    self.Gp1_LVK = lambdify([b, Rs], Gp1_LVK)
    
    # evaluating symbolic expressions in complementary point
    IL2_LVK = self.IL_LVK.subs(keys_ref).subs(self.zip_meas_to_ds2)
    I02_LVK = self.I0_LVK.subs(keys_ref).subs(self.zip_meas_to_ds2)
    Gp2_LVK = self.Gp_LVK.subs(keys_ref).subs(self.zip_meas_to_ds2)
    self.IL2_LVK = lambdify([b, Rs], IL2_LVK)
    self.I02_LVK = lambdify([b, Rs], I02_LVK)
    self.Gp2_LVK = lambdify([b, Rs], Gp2_LVK)
    
    if self.reg_monitor:
        # evaluating symbolic expressions in regular...... point
        IL3_LVK = self.IL_LVK.subs(keys_ref).subs(self.zip_meas_to_ds3)
        I03_LVK = self.I0_LVK.subs(keys_ref).subs(self.zip_meas_to_ds3)
        Gp3_LVK = self.Gp_LVK.subs(keys_ref).subs(self.zip_meas_to_ds3)
        self.IL3_LVK = lambdify([b, Rs], IL3_LVK)
        self.I03_LVK = lambdify([b, Rs], I03_LVK)
        self.Gp3_LVK = lambdify([b, Rs], Gp3_LVK)
    
    # dP/dV = 0
    F1 = self.F0.subs(self.zip_meas_to_ds1)
    F2 = self.F0.subs(self.zip_meas_to_ds2)
    self.F1 = lambdify([b, I0, Rs, Gp], F1)
    self.F2 = lambdify([b, I0, Rs, Gp], F2)
    if self.reg_monitor:
        R1 = self.F0.subs(self.zip_meas_to_ds3)
        self.R1 = lambdify([b, I0, Rs, Gp], R1)
    
    # Another equation: equal IL in both operating conditions
    # clear ILref in both conditions
    b1, IL1, I01, Rs1, Gp1 = self.params1
    b2, IL2, I02, Rs2, Gp2 = self.params2
    
    IL1_sym = self.modelSym.ILfun(self.S1, self.T1)
    IL2_sym = self.modelSym.ILfun(self.S2, self.T2)
    
    ILref1 = solve(Eq(IL1, IL1_sym), IL_ref)[0].subs(keys_m1)
    ILref2 = solve(Eq(IL2, IL2_sym), IL_ref)[0].subs(keys_m2)
    
    F3 = (ILref1 - ILref2).subs({IL1:IL1_LVK.subs({b:b1, Rs:Rs1}), IL2:IL2_LVK.subs({b:b2, Rs:Rs2})})
    self.F3 = lambdify([b1, b2, Rs1, Rs2, alphaT], F3)
    
    if self.reg_monitor:
        b3, IL3, I03, Rs3, Gp3 = self.params3
        IL3_sym = self.modelSym.ILfun(self.S3, self.T3)
        ILref3 = solve(Eq(IL3, IL3_sym), IL_ref)[0].subs(keys_m3)
        R2 = (ILref1 - ILref3).subs({IL1:IL1_LVK.subs({b:b1, Rs:Rs1}), IL3:IL3_LVK.subs({b:b3, Rs:Rs3})})
        self.R2 = lambdify([b1, b3, Rs1, Rs3, alphaT], R2)


  def eval_params(self, ref_params, S_params, T_params, S, T):
    b1, IL1, I01, Rs1, Gp1 = ref_params
    mI0, mRs, mGp = S_params
    alphaT, deltaI0, deltaRs, gammaImp, gammaVmp = T_params
      
    b  = self.bfun(b1, T)
    IL = self.ILfun(IL1, alphaT, S, T)
    I0 = self.I0fun(I01, deltaI0, mI0, S, T)
    Rs = self.Rsfun(Rs1, deltaRs, mRs, S, T)
    Gp = self.Gpfun(Gp1, mGp, S)
    return [b, IL, I0, Rs, Gp]

  def eval_eqs(self, x, verbose=False, return_params=True):   
    # unpacking variables
    b1, Rs1, Rs2 = x

    # compute parameters for both operating conditions      
    IL1 = self.IL1_LVK(b1, Rs1)
    I01 = self.I01_LVK(b1, Rs1)
    Gp1 = self.Gp1_LVK(b1, Rs1)
    params1 = [b1, IL1, I01, Rs1, Gp1]

    S2, T2 = self.meas_ds2[:2]      
    b2 = self.bfun(b1, T2)
    IL2 = self.IL2_LVK(b2, Rs2)
    I02 = self.I02_LVK(b2, Rs2)
    Gp2 = self.Gp2_LVK(b2, Rs2)
    params2 = [b2, IL2, I02, Rs2, Gp2]
    
    # compute temperature coefficients
    T_params = self.compute_temperature_coefficients(params1)
    alphaT, deltaI0, deltaRs = T_params[:3]

    # compute irradiance coefficients
    [mI0, mRs, mGp] = self.compute_irradiance_coefficients(params1, params2, T_params)
    S_params = [mI0, mRs, mGp]
    
    # dPi/dV = 0
    eF1 = self.F1(b1, I01, Rs1, Gp1)
    eF2 = self.F2(b2, I02, Rs2, Gp2)

    # IL_ref(1) = IL_ref(2)
    eF3 = self.F3(b1, b2, Rs1, Rs2, alphaT)

    if self.reg_monitor:
        # Eval in third condition
        S3, T3 = self.meas_ds3[:2]
        params3 = self.eval_params(params1, S_params, T_params, S3, T3)
        [b3, IL3, I03, Rs3, Gp3] = params3
        
        eR1 = self.R1(b3, I03, Rs3, Gp3)
        eR2 = self.R2(b1, b3, Rs1, Rs3, alphaT)
    else:
        params3 = np.ones(5)*np.nan
        eR1 = np.nan
        eR2 = np.nan

    # penalty to ensure positivity of the parameters
    ePlus1 = sum([np.nanmax([-k, 0]) for k in params1])
    ePlus2 = sum([np.nanmax([-k, 0]) for k in params2])
    ePlus3 = sum([np.nanmax([-k, 0]) for k in params3])
    ePlus  = ePlus1 + ePlus2 + ePlus3

    # correlation coefficient magnitude penalization
    eS = sum([ max(abs(m)-self.mS_max, 0) for m in S_params])
    eT = sum([ max(abs(m)-self.mT_max, 0) for m in T_params[:3]])

    # all errors
    epsF  = [eF1, eF2, eF3, eR1, eR2, ePlus, eS, eT]
      
    if verbose:
        print()
        print('Parameters in operations points')
        print("   | {:^9s} | {:^9s} | {:^9s} |".format(self.pto1, self.pto2, self.pto3))
        print(' b | {:>9.4f} | {:>9.4f} | {:>9.4f} |'.format(params1[0], params2[0], params3[0]))
        print('IL | {:>9.4f} | {:>9.4f} | {:>9.4f} |'.format(params1[1], params2[1], params3[1]))
        print('I0 | {:>7.3e} | {:>7.3e} | {:>7.3e} |'.format(params1[2], params2[2], params3[2]))
        print('Rs | {:>9.4f} | {:>9.4f} | {:>9.4f} |'.format(params1[3], params2[3], params3[3]))
        print('Gp | {:>9.4f} | {:>9.4f} | {:>9.4f} |'.format(params1[4], params2[4], params3[4]))

        print()
        print('T-related coefficients')
        print('alphaT:   {:6.4e}'.format(alphaT))
        print('deltaI0:  {:6.4e}'.format(deltaI0))
        print('deltaRs:  {:6.4e}'.format(deltaRs))

        print()
        print('S-related coefficients')
        print('mI0: {:6.4f}'.format(mI0))
        print('mRs: {:6.4f}'.format(mRs))
        print('mGp: {:6.4f}'.format(mGp))

        print()
        print('constraints')
        print('F1: {:>7.3e}'.format(eF1))
        print('F2: {:>7.3e}'.format(eF2))
        print('F3: {:>7.3e}'.format(eF3))

        print()
        print('regulation')
        print('R1: {:>7.3e}'.format(eR1))
        print('R2: {:>7.3e}'.format(eR2))
        

    if return_params:
        # data structure
        params = {
          'S_ref':self.meas_ds1[0],
          'T_ref':self.meas_ds1[1],
          'b_ref':  params1[0],
          'IL_ref': params1[1],
          'I0_ref': params1[2],
          'Rs_ref': params1[3],
          'Gp_ref': params1[4],
          'mI0': mI0, 
          'mRs': mRs, 
          'mGp': mGp,
          'alphaT': alphaT, 
          'deltaI0': deltaI0, 
          'deltaRs': deltaRs,
          }    
        return params, epsF
    else:
        return epsF


  def get_params(self, x0, tol=1e-9):
    """
    minimize F(x) = 0.5 * sum( w_i*loss(F_1(x)**2), i = 0, ..., m - 1)
    subject to   b1, Rs1, Rs2 = x
                b1_min <= b1  <= b1_max
               Rs1_min <= Rs1 <= Rs1_max(b1)
               Rs2_min <= Rs2 <= Rs2_max(b1)
    """
    assert isinstance(x0, list)
    assert isinstance(tol, float)
    columns = ['n_iter', 'b1_0', 'Rs1_0', 'Rs2_0', 'eF1', 'eF2', 'eF3', 'eR1', 'eR2', 'ePlus', 'eS', 'eT']
    print("{:^8s} | {:^8s} {:^8s} {:^8s} | {:^9s} {:^9s} {:^9s} | {:^9s} {:^9s} | {:^9s} {:^9s} {:^9s} |".format(*columns))
    print("{:>8d} | {:>8.4f} {:>8.4f} {:>8.4f} |".format(0, *x0), end='\n')

    vars_iter = []
    def funObj(x):
      # unpacking numeric variables
      b1, Rs1, Rs2 = x
      # with io.capture_output() as captured:
      if True:
        epsF = self.eval_eqs(x, verbose=False, return_params=False)
        eF1, eF2, eF3, eR1, eR2, ePlus, eS, eT = epsF 
        vars_iter.append([b1, Rs1, Rs2,*self.loss(np.square(epsF))])
        
      str_format = "{:>8d} | {:>8.4f} {:>8.4f} {:>8.4f} | {:>6.3e} {:>6.3e} {:>6.3e} | {:>6.3e} {:>6.3e} | {:>6.3e} {:>6.3e} {:>6.3e} |"
      print((str_format+' '*50).format(len(vars_iter), *vars_iter[-1]), end='\r')
      if self.regularizer:
          if self.regularizer_mS and self.regularizer_mT:
              return [eF1, eF2, eF3,
                      eR1, eR2, 
                      ePlus, eS, eT]
          elif self.regularizer_mS:
              return [eF1, eF2, eF3,
                      eR1, eR2, 
                      ePlus, eS]
          elif self.regularizer_mT:
              return [eF1, eF2, eF3,
                      eR1, eR2, 
                      ePlus, eT]
          else:
              return [eF1, eF2, eF3,
                      eR1, eR2, 
                      ePlus]
      else:
          return [eF1, eF2, eF3]
          
    
    
    try:
        root = least_squares( 
            fun = funObj,
            x0=x0, xtol=tol, ftol=tol, gtol=tol, 
            bound_callbacks = lambda x: self.bound_callbacks(x, iterative=True),
            loss=lambda z: self.loss(z, only_cost=False),
            weigths = self.weigths,
            method='trf', jac='2-point',    
            )
        print()
        vars_iter=  np.array(vars_iter).astype(np.float32)
        params_11PFF, epsF = self.eval_eqs(root.x, verbose=True)
        df = pd.DataFrame(vars_iter, columns= columns[1:])  
        root_x = root.x
        root_fun = np.array(epsF)
    except Exception as e:
        print()
        print(e)
        vars_iter = np.array(vars_iter).astype(np.float32)
        df = pd.DataFrame(vars_iter, columns= columns[1:])  
        root_x = df.iloc[-1:][['b','Rs1', 'Rs2']].values[0]
        params_11PFF, epsF = self.eval_eqs(root_x, verbose=True)
        root_fun = np.array(epsF)


    dfx = pd.DataFrame([[df.shape[0]]+ root_x.tolist() + root_fun.tolist()+ list(params_11PFF.values())],
                        columns = columns + list(params_11PFF.keys()))

    return dfx, df


  def solve_grid(self, b_arr, Rs1_arr, Rs2_arr):
    def run_task(row):
        columns=['eF1', 'eF2', 'eF3', 'eR1', 'eR2', 'ePlus', 'eS', 'eT']
        with io.capture_output() as captured:        
        # if True:
          try:
            epsF = self.eval_eqs(row, verbose=False, return_params=False)
          except:
            epsF = np.ones(8)*np.nan
        
        for id_col, col in enumerate(columns):
            row[col] = epsF[id_col]
        return row
    
    T_ref = self.meas_ds1[1]
      
    # to grid  
    b_grid, Rs1_grid, Rs2_grid = np.meshgrid(b_arr, Rs1_arr, Rs2_arr, indexing='ij')
    assert np.all(b_grid[:,0,0]   == b_arr)
    assert np.all(Rs1_grid[0,:,0] == Rs1_arr)
    assert np.all(Rs2_grid[0,0,:] == Rs2_arr)
    
    # grid to vector
    stack3d = np.vstack([b_grid.ravel(), Rs1_grid.ravel(), Rs2_grid.ravel()])
    df_grid = pd.DataFrame(stack3d.T, columns=['b1', 'Rs1', 'Rs2'])
    
    # get limits
    vRs_limit = np.vectorize(self.Rs_limits)
    Rs1min, Rs1max = vRs_limit(df_grid.b1, T_ref, self.meas_ds1[1], self.zip_meas_to_ds1)
    Rs2min, Rs2max = vRs_limit(df_grid.b1, T_ref, self.meas_ds2[1], self.zip_meas_to_ds1)
    
    # get mask
    Rs1_fil = (Rs1min<=df_grid.Rs1)&(df_grid.Rs1<=Rs1max)
    Rs2_fil = (Rs2min<=df_grid.Rs2)&(df_grid.Rs2<=Rs2max)
    
    # compute points
    df_grid2 = df_grid.loc[Rs1_fil&Rs2_fil].swifter.apply(run_task, axis = 1)
    df_grid[['eF1', 'eF2', 'eF3', 'eR1', 'eR2', 'ePlus', 'eS', 'eT']] = np.nan
    df_grid.loc[df_grid2.index] = df_grid2
    
    
    index = np.logical_not(df_grid.iloc[:, 3:].isna().any(axis=1))
    df_grid.loc[index,['loss']] = self.loss(df_grid.loc[index,['eF1', 'eF2', 'eF3']].pow(2)).sum(axis=1)
    
        
    if self.regularizer_mS and self.regularizer_mT:
        loss_reg = self.loss(df_grid.loc[index,['eR1', 'eR2', 'ePlus', 'eS', 'eT']].pow(2)).sum(axis=1)
    elif self.regularizer_mS:
        loss_reg = self.loss(df_grid.loc[index,['eR1', 'eR2', 'ePlus', 'eS']].pow(2)).sum(axis=1)
    elif self.regularizer_mT:
        loss_reg = self.loss(df_grid.loc[index,['eR1', 'eR2', 'ePlus', 'eT']].pow(2)).sum(axis=1)
    else:
        loss_reg = self.loss(df_grid.loc[index,['eR1', 'eR2', 'ePlus']].pow(2)).sum(axis=1)
    
    df_grid.loc[index,['loss_reg']] = self.w*df_grid.loc[index, 'loss']+(1-self.w)*loss_reg        
    return df_grid 


  def get_initial_guess(self, n_pts:int=20, use_presolve:bool=True):
    # global arrays
    b1_arr  = np.linspace(self.lb[0], self.ub[0], n_pts)
    Rs1_arr = np.linspace(self.lb[1], self.ub[1], n_pts)
    Rs2_arr = np.linspace(self.lb[2], self.ub[2], n_pts)
    if use_presolve:
        print('presolve')
        df_grid = self.solve_grid(b1_arr, Rs1_arr, Rs2_arr)

        # get subregion limits
        b1_min, Rs1_min, Rs2_min = df_grid.loc[:, ['b1', 'Rs1', 'Rs2']].min()
        b1_max, Rs1_max, Rs2_max = df_grid   .loc[:, ['b1', 'Rs1', 'Rs2']].max()

        # sub regions arrays
        b1_arr2   = np.linspace(b1_min, b1_max, n_pts)
        Rs1_arr2 = np.linspace(Rs1_min, Rs1_max, n_pts)
        Rs2_arr2 = np.linspace(Rs2_min, Rs2_max, n_pts)

        print('solve')
        return self.solve_grid(b1_arr2, Rs1_arr2, Rs2_arr2)
        
    else:
        print('solve')
        return self.solve_grid(b1_arr, Rs1_arr, Rs2_arr)






















