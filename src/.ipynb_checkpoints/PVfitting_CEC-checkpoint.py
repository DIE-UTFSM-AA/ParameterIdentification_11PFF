from itertools import product
import copy
import pandas as pd 
import numpy as np
from time import perf_counter
from IPython.utils import io
from scipy.interpolate import interp1d
from least_squares_dynamics_bounds.least_squares import least_squares
from sympy import symbols, lambdify
from scipy.constants import zero_Celsius as T0
from PVfitting import PVfitting
eps = np.finfo(float).eps

from src.PVModel import Model11PFF, Model7PFF, Model6PFF, Model5PFF
from sympy import solve, Eq
from scipy import interpolate
from IPython.display import clear_output
import swifter


class PVfitting_11PFF_CEC(PVfitting):
  S1, T1 = symbols("S_{ref1}, T_{ref1}", real=True)
  S2, T2 = symbols("S_{ref2}, T_{ref2}", real=True)
  S3, T3 = symbols("S_{ref3}, T_{ref3}", real=True)
  
  meas1 = symbols("V_{oc-1}, I_{sc-1}, V_{mp-1}, I_{mp-1}", positive=True, real=True)
  meas2 = symbols("V_{mp-2}, I_{mp-2}", positive=True, real=True)
  meas3 = symbols("V_{mp-3}, I_{mp-3}", positive=True, real=True)

  @property    
  def zip_meas_to_ds1(self):  
    return dict(zip(self.meas, self.meas_ds1[2:-1]))

  @property
  def zip_meas_to_ds2(self):
    return dict(zip(self.meas[2:], self.meas_ds2[2:-1]))

  @property
  def zip_meas_to_ds3(self):
    return dict(zip(self.meas[2:], self.meas_ds3[2:-1]))

  @property    
  def zip_meas1_to_ds1(self):  
    return dict(zip([self.S1]+ [self.T1] + list(self.meas1), self.meas_ds1[:-1]))

  @property    
  def zip_meas2_to_ds2(self):  
    return dict(zip([self.S2]+ [self.T2] + list(self.meas2), self.meas_ds2[:-1]))

  @property    
  def zip_meas3_to_ds3(self):  
    return dict(zip([self.S3]+ [self.T3] + list(self.meas3), self.meas_ds3[:-1]))




  def bound_callbacks(self, bref, iterative=False):
    T_ref = self.meas_ds1[1] 
    Rs_ref1_lb, Rs_ref1_ub = self.Rs_limits(bref, T_ref, T_ref, self.zip_meas_to_ds1)
    if iterative:
      lb = [self.bref_lb, Rs_ref1_lb, -np.inf, -np.inf, -np.inf]
      ub = [self.bref_ub, Rs_ref1_ub,  np.inf,  np.inf,  np.inf]
      
      lb = [self.bref_lb, Rs_ref1_lb, -self.mS_max, -self.mS_max, -self.mS_max]
      ub = [self.bref_ub, Rs_ref1_ub,  self.mS_max, self.mS_max, self.mS_max]


    else:
      lb = [self.bref_lb, Rs_ref1_lb]
      ub = [self.bref_ub, Rs_ref1_ub]
    return lb, ub

  def __init__(self, df_summary):
    self.df_summary = copy.deepcopy(df_summary)
    df_datasheet = self.df_summary.iloc[:, self.df_summary.columns.get_level_values(0).str.contains('DataSheet')]
    self.df_datasheet = df_datasheet.reset_index().set_index('Module')  
    DAta = self.df_datasheet.iloc[:,self.df_datasheet.columns.get_level_values(1).str.contains('STC|NOCT|LIC')]
    DAta = DAta.T.droplevel(0).T
    DAta.iloc[:,DAta.columns.get_level_values(1).str.contains('T')]+=T0

    Tcoeff = self.df_datasheet.iloc[:,self.df_datasheet.columns.get_level_values(1).str.contains('alpha|beta|gamma')]
    Tcoeff = Tcoeff.T.droplevel(0).droplevel(1).T
    
    Ns_Np = self.df_datasheet.iloc[:,self.df_datasheet.columns.get_level_values(1).str.contains('N_s|N_p')]
    Ns_Np = Ns_Np.T.droplevel(0).droplevel(1).T

    no_converge = []
    with open('src/ui/no_converge.txt', 'r') as f:
      for line in f:
        no_converge.append(line.strip())
    self.no_converge0 = np.unique(no_converge).tolist()

    # only for this instance {CEC or NREL)
    self.no_converge  = list(set(self.df_datasheet.index.to_list())&set(self.no_converge0))

    # external elements for this instance 
    # if work in CEC, this list has PV modules at NREL
    self.no_converge0 = list(set(self.no_converge0)-set(self.no_converge))
    PVfitting.__init__(self, DAta, Tcoeff, Ns_Np)

    
     
  
  def compile(self, PVModule, Ns, n_max=2.5, n_min=0.5, verbose=True):
    self.pto1 = 'STC'
    self.pto2 = 'NOCT' 
    self.pto3 = 'LIC'

    # =========================================================
    # Load measurement and T_coefficients from datasheet
    # =========================================================
    # temperatura coefficients
    self.spec_Tcoeff = self.get_temperature_coefficients(PVModule, self.Tcoeff)
      
    data = self.DAta.loc[[PVModule]]

    # get first point (reference)
    self.meas_ds1 = data.iloc[:,data.columns.get_level_values(0).str.contains(self.pto1)].T.droplevel(0).T.values[0].tolist()
    S_ref, T_ref = self.meas_ds1[:2]
    S1, T1 = self.meas_ds1[:2]

    # get second point (complementary 1)
    self.meas_ds2 = data.iloc[:,data.columns.get_level_values(0).str.contains(self.pto2)].T.droplevel(0).T.values[0].tolist()
    S2, T2 = self.meas_ds2[:2]

    # get second point (complementary 2)
    self.meas_ds3 = data.iloc[:,data.columns.get_level_values(0).str.contains(self.pto3)].T.droplevel(0).T.values[0].tolist()
    S3, T3 = self.meas_ds3[:2]

    if verbose:
      self.print_point(self.pto1, self.meas_ds1, 'Reference')
      self.print_point(self.pto2, self.meas_ds2, 'Complementary 1')
      self.print_point(self.pto3, self.meas_ds3, 'Complementary 2')
    
    # 1/2: [Vmp, Imp] -> drop nan (Voc, Isc)
    del self.meas_ds2[2]
    del self.meas_ds2[2]
    
    # 1/2: [Vmp, Imp] -> drop nan (Voc, Isc)
    del self.meas_ds3[2]
    del self.meas_ds3[2]

    # =========================================================
    # Defined global bounds
    # =========================================================
    print()
    print('global bounds')
    [self.bref_lb, self.bref_ub] = self.b_limits(T_ref, n_max, n_min, Ns)
    self.lb, self.ub = self.bound_callbacks(self.bref_lb)
    print('lb: {0}'.format(self.lb))
    print('ub: {0}'.format(self.ub))  
    assert (np.array(self.ub)>np.array(self.lb)).all()

    
    # =================================================
    # unpacking variables and dictionaries
    # =================================================
    b, IL, I0, Rs, Gp = self.params
    b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref = self.params_ref
    alphaT, deltaI0, deltaRs = self.T_coefficients
    mI0, mRs, mGp = self.S_coefficients

    # all keys
    keys_ref = {self.Sref:S_ref, self.Tref:T_ref}

    # =================================================
    # symbolic temperature system 
    # =================================================
    self.A_T1 = self.A_temp.subs(self.zip_meas_to_ds1).subs(keys_ref)
    self.b_T1 = self.b_temp.subs(self.zip_meas_to_ds1).subs(keys_ref)

    
    # =================================================
    # 11pff functions
    # =================================================
    self.bfun = self.modelSym11PFF.bfun(self.T).subs(keys_ref)
    self.bfun = lambdify([b_ref, self.T], self.bfun)
    
    self.ILfun = self.modelSym11PFF.ILfun(self.S, self.T).subs(keys_ref)
    self.ILfun = lambdify([IL_ref, alphaT, self.S, self.T], self.ILfun)
    
    self.I0fun = self.modelSym11PFF.I0fun(self.S, self.T).subs(keys_ref)
    self.I0fun = lambdify([I0_ref, mI0, deltaI0, self.S, self.T], self.I0fun)
    
    self.Rsfun = self.modelSym11PFF.Rsfun(self.S, self.T).subs(keys_ref)
    self.Rsfun = lambdify([Rs_ref, mRs, deltaRs, self.S, self.T], self.Rsfun)
    
    self.Gpfun = self.modelSym11PFF.Gpfun(self.S).subs(keys_ref)
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
    
    # solve (dP/dV = 0)
    F01 = self.F0.subs(self.zip_meas_to_ds1)
    F02 = self.F0.subs(self.zip_meas_to_ds2)
    F03 = self.F0.subs(self.zip_meas_to_ds3)
    self.F01 = lambdify([b, I0, Rs, Gp], F01)
    self.F02 = lambdify([b, I0, Rs, Gp], F02)
    self.F03 = lambdify([b, I0, Rs, Gp], F03)
    
    # LVK (A[0] - A[1] = 0)
    get_MPP_LVK = self.get_MPP_LVK.args[0] - self.get_MPP_LVK.args[1]
    LVK2 = get_MPP_LVK.subs(self.zip_meas_to_ds2)
    LVK3 = get_MPP_LVK.subs(self.zip_meas_to_ds3)
    self.LVK2 = lambdify([b, IL, I0, Rs, Gp], LVK2)
    self.LVK3 = lambdify([b, IL, I0, Rs, Gp], LVK3)

  def eval_params(self, ref_params, S_params, T_params, S, T):
    b1, IL1, I01, Rs1, Gp1 = ref_params
    mI0, mRs, mGp = S_params
    alphaT, deltaI0, deltaRs, gammaImp, gammaVmp = T_params
      
    b  = self.bfun(b1, T)
    IL = self.ILfun(IL1, alphaT, S, T)
    I0 = self.I0fun(I01, mI0, deltaI0, S, T)
    Rs = self.Rsfun(Rs1, mRs, deltaRs, S, T)
    Gp = self.Gpfun(Gp1, mGp, S)
    return [b, IL, I0, Rs, Gp]

  def eval_eqs(self, x, verbose=False, return_params=True):
    lb, ub = self.bound_callbacks(x[0], iterative=True)
    lb = np.real(lb)
    ub = np.real(ub)
    f_lb = x<lb
    f_ub = x>ub
    x[f_ub] = ub[f_ub]
    x[f_lb] = lb[f_lb]
      
    # unpacking variables
    b1, Rs1, mI0, mRs, mGp = x
    S_params = [mI0, mRs, mGp]
    S_ref, T_ref = self.meas_ds1[:2]
    
    # compute reference params
    IL1 = self.IL1_LVK(b1, Rs1)
    I01 = self.I01_LVK(b1, Rs1)
    Gp1 = self.Gp1_LVK(b1, Rs1)
    params1 = [b1, IL1, I01, Rs1, Gp1]

    # compute temperature coefficients
    T_params = self.compute_temperature_coefficients(params1)
    
    
    # Eval in second condition
    S2, T2 = self.meas_ds2[:2]
    params2 = self.eval_params(params1, S_params, T_params, S2, T2)
    [b2, IL2, I02, Rs2, Gp2] = params2
    
    # Eval in third condition
    S3, T3 = self.meas_ds3[:2]
    params3 = self.eval_params(params1, S_params, T_params, S3, T3)
    [b3, IL3, I03, Rs3, Gp3] = params3

    # penalty to ensure positivity of the parameters
    ePlus1 = sum([np.nanmax([-k, 0]) for k in params1])
    ePlus2 = sum([np.nanmax([-k, 0]) for k in params2])
    ePlus3 = sum([np.nanmax([-k, 0]) for k in params3])
    ePlus  = ePlus1 + ePlus2 + ePlus3
    
    # eps dPi/dV = 0
    eF01 = self.F01(b1, I01, Rs1, Gp1)
    eF02 = self.F02(b2, I02, Rs2, Gp2)
    eF03 = self.F03(b3, I03, Rs3, Gp3)

    # eps LVK
    eLVK2 = self.LVK2(b2, IL2, I02, Rs2, Gp2)
    eLVK3 = self.LVK3(b3, IL3, I03, Rs3, Gp3)

    # all eps
    eps  = [eF01, eF02, eF03, eLVK2, eLVK3, ePlus]


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
        print('constraints')
        print('F01: {:>7.3e}'.format(eF01))
        print('F02: {:>7.3e}'.format(eF02))
        print('F03: {:>7.3e}'.format(eF03))

        print()
        print('LVK')
        print('LVK2: {:>7.3e}'.format(eLVK2))
        print('LVK3: {:>7.3e}'.format(eLVK3))

    

    if return_params:
        # data structure
        params = {
          'S_ref': S_ref,
          'T_ref': T_ref,
          'b_ref': params1[0],
          'IL_ref': params1[1],
          'I0_ref': params1[2],
          'Rs_ref': params1[3],
          'Gp_ref': params1[4],
          'mI0': mI0, 
          'mRs': mRs, 
          'mGp': mGp,
          'alphaT': T_params[0], 
          'deltaI0': T_params[1], 
          'deltaRs': T_params[2]
          }  
        
        return params, eps
    else:
        return eps

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
    columns = ['n_iter', 'b1_0', 'Rs1_0', 'mI0_0', 'mRs_0', 'mGp_0', 'eF1', 'eF2', 'eF3', 'eF4', 'eF5', 'ePlus']
    print("{:^8s} | {:^8s} {:^8s} {:^8s} {:^8s} {:^8s} | {:^9s} {:^9s} {:^9s} {:^9s} {:^9s} | {:^9s} |".format(*columns))
    print("{:>8d} | {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} |".format(0, *x0), end='\n')

    vars_iter = []
    def funObj(x):
      # unpacking numeric variables
      b1, Rs1, mI0, mRs, mGp = x
      # with io.capture_output() as captured:
      if True:
        epsF = self.eval_eqs(x, verbose=False, return_params=False)
        eF01, eF02, eF03, eLVK2, eLVK3, ePlus = epsF 
        vars_iter.append([b1, Rs1, mI0, mRs, mGp, *self.loss(np.square(epsF))])
        
      str_format = "{:>8d} | {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} | {:>6.3e} {:>6.3e} {:>6.3e} {:>6.3e} {:>6.3e} | {:>6.3e} |"
      print((str_format+' '*50).format(len(vars_iter), *vars_iter[-1]), end='\r')
      if epsF[-1]>0:
        return np.nan
      else:
        return epsF[:-1]
          
    try:
      root = least_squares( 
              fun = funObj,
              x0=x0, xtol=tol, ftol=tol, gtol=tol, 
              bound_callbacks = lambda x: self.bound_callbacks(x, iterative=True),
              # loss=lambda z: self.loss(z, only_cost=False),
              loss = 'cauchy',
              weigths = [1, 1, 1, 1, 1],
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


    dfx = pd.DataFrame([[df.shape[0]]+ x0 + root_fun.tolist()+ list(params_11PFF.values())],
                        columns = columns + list(params_11PFF.keys()))

    return dfx, df


  def get_initial_guess_7pff(self, pts=10, tol=1e-9):
    S_ref, T_ref = self.meas_ds1[:2]
    S1, T1 = self.meas_ds1[:2]
  
    # fixed coefficients
    alpha = self.spec_Tcoeff[0]
    deltaI0 = 1
    mRs = 0
    mGp = 1
    
    columns = ['n_iter', 'b1_0', 'Rs1_0', 'mI0_0', 'eFfp', 'eFtc', 'eFtc2', 'ePlus']
    print("{:^8s} | {:^8s} {:^8s} {:^8s} | {:^9s} {:^9s} {:^9s} | {:^9s} |".format(*columns))
    vars_iter = []
      
    def funObj(x, grid=False):
      b1, Rs1, mI0 = x
  
      # compute reference params
      IL1 = self.IL1_LVK(b1, Rs1)
      I01 = self.I01_LVK(b1, Rs1)
      Gp1 = self.Gp1_LVK(b1, Rs1)
      params1 = [b1, IL1, I01, Rs1, Gp1]
      
      if grid:
        try:
          # compute temperature coefficients
          T_params = self.compute_temperature_coefficients(params1)
          alphaT1, deltaI01, deltaRs1 = T_params[:3]            
      
          # compute loss
          Ffp = self.F01(b1, I01, Rs1, Gp1)
          Ftc = deltaI01/deltaI0-1
          Ftc2 = alphaT1/alpha-1
          ePlus = sum([np.nanmax([-k, 0]) for k in params1])
          
          if ePlus>0:
            return [np.nan, np.nan, np.nan, np.nan]
          else:
            return [Ffp, Ftc, Ftc2, ePlus]
        except:
          return [np.nan, np.nan, np.nan, np.nan]
      else:
        # compute temperature coefficients
        T_params = self.compute_temperature_coefficients(params1)
        alphaT1, deltaI01, deltaRs1 = T_params[:3]            
    
        # compute loss
        Ffp = self.F01(b1, I01, Rs1, Gp1)
        Ftc = deltaI01/deltaI0-1
        Ftc2 = alphaT1/alpha-1
        ePlus = sum([np.nanmax([-k, 0]) for k in params1])
        epsF = [Ffp, Ftc, Ftc2, ePlus]

        vars_iter.append([b1, Rs1, mI0, *self.loss(np.square(epsF))])
        
        str_format = "{:>8d} | {:>8.4f} {:>8.4f} {:>8.4f} | {:>6.3e} {:>6.3e} {:>6.3e} | {:>6.3e} |"
        print((str_format+' '*50).format(len(vars_iter), *vars_iter[-1]), end='\r')
        if ePlus>0:
          return [np.nan, np.nan, np.nan, np.nan]
        else:
          return [Ffp, Ftc, Ftc2, ePlus]
        
    def bound_callbacks(x):
        lb, ub = self.bound_callbacks(x, iterative=True)
        return lb[:3], ub[:3]

    time0 = perf_counter()
    vRs_limit = np.vectorize(self.Rs_limits)
    b1min = self.lb[0]
    b1max = self.ub[0]
    b1 = np.linspace(b1min, b1max, pts)
    
    Rs1min, Rs1max = vRs_limit(b1, T_ref, T1, self.zip_meas_to_ds1)
    Rs1min = Rs1min.min()
    
    interp_func = interp1d(Rs1max, b1, kind='linear', fill_value='extrapolate')
    bRs1min = interp_func(Rs1min)
        
    # grid of optimal solution
    xb1 = np.linspace(bRs1min, b1max, pts)
    xb1 = xb1[xb1>0]
    xb1 = xb1[xb1>b1min]
    xRs1min, xRs1max = vRs_limit(xb1, T_ref, T1, self.zip_meas_to_ds1)
    xRs1max = xRs1max[xRs1max>0]
    
    # eval in 7PFF (F1 y F2)        
    all_results = []
    for k in range(pts):
        Rs1 = Rs1min + (xRs1max-Rs1min)*k/(pts-1)
        x_arr = list(product(xb1, Rs1, [1]))
        z = []
        for xx in x_arr:
          with io.capture_output() as captured:
            z.append(
                self.loss(
                  np.square(
                    funObj(xx, grid=True)
                    )
                  )
            )
        z = np.array(z)
        z = np.log(z[:,0]**2+z[:,1]**2)
        zz = np.concatenate([np.array(x_arr), np.expand_dims(z, 1)], axis=1)  
        all_results.append(zz)

    all_results = np.concatenate(all_results)
    filter_Rs = all_results[:, 1]<vRs_limit(all_results[:, 0], T_ref, T1, self.zip_meas_to_ds1)[1]
    all_results = all_results[filter_Rs]

    x0 = all_results[all_results[:,-1].argmin(),:-1]
    loss_x0 = self.loss(np.square(funObj(x0, grid=True)))
    str_format = "{:>8d} | {:>8.4f} {:>8.4f} {:>8.4f} | {:>6.3e} {:>6.3e} {:>6.3e} | {:>6.3e} |"
    print(str_format.format(0, *x0, *loss_x0), end='\n')
    
    # =====================================================================
    # add fsolve to improve initial estimation performance
    root = least_squares( 
        fun = funObj,
        x0=x0, 
        xtol=tol, ftol=tol, gtol=tol, 
        bound_callbacks = bound_callbacks,
        # loss=lambda z: self.loss(z, only_cost=False),
        loss = 'cauchy',
        weigths = [1, 1, 1, 1],
        method='trf', jac='2-point'
      )
    time1 = perf_counter()
    b_opt, Rs1_opt, mI0_opt = root.x

    k7PFF = [
        'n_iter',
        'converge',
        'b1_0', 'Rs1_0', 'mI0_0',
        'Ffp', 'Ftc', 'Ftc2', 'ePlus',
        'S_ref', 'T_ref',
        'b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
        'mI0', 'mRs', 'mGp',
        'alphaT', 'deltaI0', 'deltaRs', 
             ]

    converge = root.success
    n_iter = len(vars_iter)
    [Ffp, Ftc, Ftc2, ePlus] = funObj(root.x)
    
    IL1 = self.IL1_LVK(b_opt, Rs1_opt)
    I01 = self.I01_LVK(b_opt, Rs1_opt)
    Gp1 = self.Gp1_LVK(b_opt, Rs1_opt)
    params1 = [b_opt, IL1, I01, Rs1_opt, Gp1]
    T_params = self.compute_temperature_coefficients(params1)
    alphaT1, deltaI01, deltaRs1 = T_params[:3]
    
    
    df7pff = pd.DataFrame([[
        n_iter, converge,
        x0[0], x0[1], x0[2],
        Ffp, Ftc, Ftc2, ePlus,
        S_ref, T_ref,
        b_opt, IL1, I01, Rs1_opt, Gp1, 
        mI0_opt, mRs, mGp,
        alpha, deltaI0, deltaRs1
    ]], columns=k7PFF)
    
    return df7pff





  def eval(self, PVModule, 
               n_max=2.0, 
               n_min=0.5, 
               Rs_min=1e-6,
               pts_presolve=10, 
               tol=1e-9,
               mS_max = 3,
              ):
    self.Rs_min = Rs_min
    Ns, Np = self.Ns_Np.loc[PVModule].astype(int)
    self.compile(PVModule, Ns, n_max=n_max, n_min=n_min)
    self.mS_max = mS_max
    
    try:
      print('initial guess: 7PFF')
      df7pff = self.get_initial_guess_7pff(pts=pts_presolve, tol=tol)             
      df7pff['Module'] = PVModule
      df7pff = df7pff.set_index(['Module'])    
      try:
        k11PFF_init = ['b_ref', 'Rs_ref', 'mI0', 'mRs', 'mGp']
        x0 = df7pff[k11PFF_init].values[0].tolist() 
        print()
        print()
        print('least_squares')
        df_sum, df_iter = self.get_params(x0, tol=tol)
        df_sum['Module'] = PVModule
        df_sum = df_sum.set_index(['Module'])    
        return df_iter, df_sum, df7pff
      except Exception as e:
        print(e)
        return None, None, df7pff
    except:
        return None, None, None
            

  def run(self, PVModule, 
                nx, ny, 
                n_max=2.0, 
                n_min=0.5, 
                Rs_min=1e-6,
                pts_presolve=10, 
                tol=1e-9,
                gtol=1e-1,
                mS_max=3.,
                n_Mmax=0.5
         ):
    def prediction(mPFF, model):
      # =========== Reference conditions
      S, T, Voc, Isc, Vmp, Imp, Pmp = self.meas_ds1
      [Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m] = mPFF.predict(S, T)
      df_STC = pd.DataFrame([[Voc_m, Isc_m, Vmp_m, Imp_m, Pmp_m]], columns=['Voc', 'Isc', 'Vmp', 'Imp', 'Pmp'])
      df_STC.columns = pd.MultiIndex.from_product([["Prediction_{0}".format(model)], ['STC'], df_STC.columns])

      # =========== Complementary conditions (NOCT)
      S, T, Vmp, Imp, Pmp = self.meas_ds2
      [Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m] = mPFF.predict(S, T)
      df_NOCT = pd.DataFrame([[Voc_m, Isc_m, Vmp_m, Imp_m, Pmp_m]], columns=['Voc', 'Isc', 'Vmp', 'Imp', 'Pmp'])
      df_NOCT.columns = pd.MultiIndex.from_product([["Prediction_{0}".format(model)], ['NOCT'], df_NOCT.columns])

      # # =========== Complementary conditions (LIC)
      S, T, Vmp, Imp, Pmp = self.meas_ds3
      [Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m] = mPFF.predict(S, T)
      df_LIC = pd.DataFrame([[Voc_m, Isc_m, Vmp_m, Imp_m, Pmp_m]], columns=['Voc', 'Isc', 'Vmp', 'Imp', 'Pmp'])
      df_LIC.columns = pd.MultiIndex.from_product([["Prediction_{0}".format(model)], ['LIC'], df_LIC.columns])
      return pd.concat([df_STC, df_NOCT, df_LIC], axis=1)
    
    def temperature_behaviour(model, dfx):
      def get_mean(df_v, df_T):
        mean = df_v.values.mean()
        c = 0
        while True:
          try:
            idx = np.argsort((df_v.values-mean)**2)[0][c]
            x_values = df_v.iloc[:,idx-2:idx+3].values[0]
            y_values = df_T.iloc[:,idx-2:idx+3].values[0]
            f = interpolate.interp1d(x_values, y_values)
            x_pred = float(f(mean))
            break
          except Exception as e:
            c+=1
          if c>df_T.shape[1]:
            break
        return x_pred, mean

      # ====================================================================
      #       system construction
      # ====================================================================
      gamma_spec = self.spec[-1]
      keys0 = {self.Sref:self.meas_ds1[0], self.S:self.meas_ds1[0], self.Tref:self.meas_ds1[1]}
      keys1 = dict(zip(self.params_ref, dfx[['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref']].values[0])) 
      keys2 = dict(zip(self.T_coefficients, dfx[['alphaT', 'deltaI0', 'deltaRs']].values[0]))
      keys3 = dict(zip(self.S_coefficients, dfx[['mI0', 'mRs', 'mGp']].values[0]))
      keys4 = dict(zip(self.meas, self.meas_ds1[2:-1]))

      dIpv_sc  = self.get_dIpv_sc(model).subs(keys0).subs(keys1).subs(keys2).subs(keys3).subs(keys4)
      dIpv_oc  = self.get_dIpv_oc(model).subs(keys0).subs(keys1).subs(keys2).subs(keys3).subs(keys4)
      dIpv_mp  = self.get_dIpv_mp(model).subs(keys0).subs(keys1).subs(keys2).subs(keys3).subs(keys4)
      F0mp_dT  = self.get_dVpv_mp(model).subs(keys0).subs(keys1).subs(keys2).subs(keys3).subs(keys4)
      dIVdT_mp = self.get_dPpv_mp(model).subs(keys0).subs(keys1).subs(keys2).subs(keys3).subs(keys4)
      
      def solution(df):
        Tx = df['T']
        df['alpha']   = float(solve(Eq(dIpv_sc.subs({self.T:Tx}),0), check=False)[0])
        df['beta']    = float(solve(Eq(dIpv_oc.subs({self.T:Tx}), 0))[0])
        gamma_imp_vmp = solve([Eq(dIpv_mp.subs({self.T:Tx}), 0), Eq(F0mp_dT.subs({self.T:Tx}), 0)])
        df['gamma']   = float(solve(Eq(dIVdT_mp.subs({self.T:Tx}), gamma_spec).subs(gamma_imp_vmp))[0])
        df['T'] -=T0 
        return df
      
      T_arr = np.arange(25, 65, 2)+T0
      df_temp = pd.DataFrame(T_arr, columns=['T'])     
      df_temp = df_temp.swifter.apply(lambda x: solution(x), axis=1)

      df_T = df_temp[['T']].T.reset_index(drop=True)
      df_T.columns = pd.MultiIndex.from_product([["Temperature_{0}".format(model)], ['T'], df_T.columns.astype(str)])

      df_alpha = df_temp[['alpha']].T.reset_index(drop=True)
      df_alpha.columns = pd.MultiIndex.from_product([["Temperature_{0}".format(model)], ['alpha'], df_alpha.columns.astype(str)])

      df_beta = df_temp[['beta']].T.reset_index(drop=True)
      df_beta.columns = pd.MultiIndex.from_product([["Temperature_{0}".format(model)], ['beta'], df_beta.columns.astype(str)])

      df_gamma = df_temp[['gamma']].T.reset_index(drop=True)
      df_gamma.columns = pd.MultiIndex.from_product([["Temperature_{0}".format(model)], ['gamma'], df_gamma.columns.astype(str)])

      alphaT, alphaM = get_mean(df_alpha, df_T)
      betaT, betaM = get_mean(df_beta, df_T)
      gammaT, gammaM = get_mean(df_gamma, df_T)

      df_temp0 = pd.DataFrame([[20]], columns=pd.MultiIndex.from_product([["Temperature_{0}".format(model)], ['Npts'], ['']]))
      df_temp1 = pd.DataFrame([[alphaT, alphaM, betaT, betaM, gammaT, gammaM]], 
                  columns=pd.MultiIndex.from_product([["Temperature_{0}".format(model)], ['alpha', 'beta', 'gamma'], ['T', 'mean']]))
      df_temp2 = pd.concat([df_temp1, df_temp0, df_T, df_alpha, df_beta, df_gamma], axis=1)
      return df_temp2

    gloss = 1e10
    g_nmax = None

    while True:
      clear_output(wait=True)
      print("{:>3d}|{:>3d}: {:s}".format(nx, ny, PVModule))
      print('n_max: {:4.2f}'.format(n_max))
      try:
        df_iter, df11pff, df7pff = self.eval(PVModule, n_max=n_max, n_min=n_min, Rs_min=Rs_min,
                                              pts_presolve=pts_presolve,tol=tol, mS_max=mS_max)
        if df11pff is None:
          pass
        else:
          df7pff_cond  = (df7pff.iloc[:, df7pff.columns.str.contains('_ref')]>0).all(axis=1).values[0]
          df11pff_cond = (df11pff.iloc[:, df11pff.columns.str.contains('_ref')]>0).all(axis=1).values[0]
          if df7pff_cond & df11pff_cond:
            loss11PFF = df11pff.loc[:,df11pff.columns.str.contains('eF')].apply(np.square).sum(axis=1)
            tol11PFF  = loss11PFF<=gtol
            if tol11PFF[0]:
              break
            else:
              if loss11PFF[0]<gloss:
                  gloss = loss11PFF[0]
                  g_nmax = n_max
          else:
            pass
      except Exception as e:
        print('Exception')
        print(e)
      n_max-=0.1
      if n_max<n_Mmax:
        break

    if not tol11PFF[0]:
      if g_nmax is not None:
        n_max = g_nmax
        clear_output(wait=True)
        print("{:>3d}|{:>3d}: {:s}".format(nx, ny, PVModule))
        print('n_max: {:4.2f}'.format(n_max))
        df_iter, df11pff, df7pff = self.eval(PVModule, n_max=n_max, n_min=n_min, Rs_min=Rs_min,
                                              pts_presolve=pts_presolve,tol=tol, mS_max=mS_max)
        

    if all([
      isinstance(df11pff, pd.DataFrame), 
      isinstance(df7pff, pd.DataFrame)]):
      print('ok')
      dfy = copy.deepcopy(df7pff)
      dfy = dfy.drop(columns=['converge'])
      lb, ub = self.bound_callbacks(dfy.b_ref[0])
      dfy.insert(0, 'converge', True)
      dfy.insert(1, 'n_max', n_max)
      dfy.insert(3, 'b_min', lb[0])
      dfy.insert(4, 'b_max', ub[0])
      dfy.insert(5, 'Rs1_min', lb[1])
      dfy.insert(6, 'Rs1_max', ub[1])
      m7PFF = Model7PFF(*dfy[self.k7PFF].values.tolist()[0])
      dfy.columns = pd.MultiIndex.from_product([['Solution_7PFF'], dfy.columns, ['']])
      df_pred7PFF = prediction(m7PFF, '7PFF')
      df_pred7PFF.index = dfy.index
      df_temp_7PFF = temperature_behaviour('7PFF', df7pff)
      df_temp_7PFF.index = dfy.index

      dfx = copy.deepcopy(df11pff)
      lb, ub = self.bound_callbacks(dfx.b_ref[0])
      dfx.insert(0, 'converge', True)
      dfx.insert(1, 'n_max', n_max)
      dfx.insert(3, 'b_min', lb[0])
      dfx.insert(4, 'b_max', ub[0])
      dfx.insert(5, 'Rs1_min', lb[1])
      dfx.insert(6, 'Rs1_max', ub[1])
      m11PFF = Model11PFF(*dfx[self.k11PFF].values.tolist()[0])
      dfx.columns = pd.MultiIndex.from_product([['Solution_11PFF'], dfx.columns, ['']])
      df_pred11PFF = prediction(m11PFF, '11PFF')
      df_pred11PFF.index = dfx.index
      df_temp_11PFF = temperature_behaviour('11PFF', df11pff)
      df_temp_11PFF.index = dfx.index
      
      df_sum = pd.concat([dfy, df_pred7PFF, df_temp_7PFF, dfx, df_pred11PFF, df_temp_11PFF], axis=1)
      df_sum0 = pd.concat([self.df_datasheet.loc[[PVModule]], df_sum], axis=1)
      df_sum0 = df_sum0.reset_index().set_index(['Technology', 'Module'])
      return df_sum0
    else:
      print(df11pff)
      print(df7pff)
    

  def __call__(self, config):
    dff = []
    ny = self.df_datasheet.shape[0]-1
    for nx, PVModule in enumerate(self.df_datasheet.index):
      if not (PVModule in self.no_converge):
        try:
          df_sum = self.run(PVModule, nx, ny, **config)
          self.df_summary.loc[df_sum.index] = df_sum
          self.df_summary.to_csv('src/ui/temp_summary.csv')
            
          dff.append(df_sum)
        except Exception as e:
          print(e)
          self.no_converge.append(PVModule)
          with open('src/ui/no_converge.txt', 'w') as f:
            for name in self.no_converge0:
                f.write(name + '\n')
            for name in self.no_converge:
                f.write(name + '\n')
          no_converge = np.unique(self.no_converge).tolist()
          pass
    return self.df_summary





