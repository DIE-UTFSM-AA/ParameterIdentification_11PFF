import copy
from sympy import exp, symbols, diff, solve, Eq, simplify, collect, linear_eq_to_matrix, lambdify
from PVModel import Model7PFF, Model11PFF
from scipy.constants import zero_Celsius as T0

class PVgeneral(object):
  # symbolic params definition
  n = symbols("n", real=True) # cell ideality coefficient
  S, T = symbols("S, T", real=True)
  Sref, Tref = symbols("S_ref, T_ref", real=True)
  Vk, Ik = symbols("V_k, I_k", real=True, positive=True)
  spec = symbols("alpha_spec, beta_spec, gamma_spec", real=True)
  meas = symbols("V_oc, I_sc, V_mp, I_mp", positive=True, real=True)
  params = symbols("b, I_L, I_0, R_s, G_p", positive=True, real=True)

  # definition in models (5 De Soto reference parameters)
  params_ref = symbols("b_ref, IL_ref, I0_ref, Rs_ref, Gp_ref", positive=True, real=True)

  # Equivalent function of a PV array
  #[b, IL, I0, Rs, Gp] = params 
  #Ipv = IL - I0*(exp(b*(Vk+Rs*Ik))-1) - Gp*(Vk+Rs*Ik)

  # proposal model coeficents (symbolic)
  S_coefficients = symbols("mI0, mRs, mGp", real=True)
  T_coefficients = symbols("alpha_T, delta_I0, delta_Rs", real=True)
  modelSym11PFF = Model11PFF(*params_ref, *S_coefficients, *T_coefficients, Tref, Sref)
  modelSym7PFF = Model7PFF(*params_ref, T_coefficients[0], S_coefficients[0], T_coefficients[2], Tref, Sref)

  # symbolic definition of derivatives
  dIk_dT = symbols(r"\dfrac{\partial~I_k}{\partial~T}", real=True)
  dVk_dT = symbols(r"\dfrac{\partial~V_k}{\partial~T}", real=True)
  dI_dV  = symbols(r"\dfrac{\partial~I}{\partial~V}", real=True)
  dIV_dV = symbols(r"\dfrac{\partial~IV}{\partial~V}", real=True)

  #dIk_dT = symbols(r"\dfrac{\partial~I_k}{\partial~T}", real=True)

  # complementary sensitivity coefficients
  gammaImp, gammaVmp = symbols("gamma_Imp, gamma_Vmp", real=True)



  def __init__(self, gamma=1.0):
    # Equivalent function of a PV array
    [b, IL, I0, Rs, Gp] = self.params 
    Vk = self.Vk
    Ik = self.Ik
    self.Ipv = IL - I0*(exp(b*(Vk+Rs*Ik))-1) - Gp*(Vk+Rs*Ik)**gamma




  def zip_sym_general2ref(self, meas_ref, params_ref):
    dict1 = dict(zip(self.meas, meas_ref))
    dict2 = dict(zip(self.params, params_ref))
    dict1.update(dict2)
    return dict1

  def get_point(self, data, pto):
    data_pto = data.loc[data.Type==pto, data.columns[1:]].values[0].tolist()
    data_pto[1] +=T0
    return data_pto

  def print_point(self, pto, data, str=''):
    # Kelvin to Celcius
    data_pto = copy.deepcopy(data)
    data_pto[1] -=T0

    print()
    print('{0} conditions ({1})'.format(pto, str))
    print('S: {:>6.1f}(W/m2) - T: {:>4.1f}(°C) - Voc: {:>5.2f}(V) - Isc: {:>5.2f}(A) - Vmp: {:>5.2f}(V) - Imp: {:>5.2f}(A) - Pmp: {:>5.2f}(W)'.format(*data_pto))     
      

        
  def Eg(self, T, Tref):
    Eg_fun = lambdify([self.T, self.Tref], self.modelSym11PFF.Eg(self.T))
    return Eg_fun(T, Tref)

  @property
  def get_OC_LVK(self):
    Voc, Isc, Vmp, Imp = self.meas
    b, IL, I0, Rs, Gp = self.params
    return Eq(self.Ipv.subs({self.Ik:0,   self.Vk:Voc}), 0)

  @property
  def get_SC_LVK(self):
    Voc, Isc, Vmp, Imp = self.meas
    b, IL, I0, Rs, Gp = self.params
    return Eq(self.Ipv.subs({self.Ik:Isc, self.Vk:0}), Isc)

  @property
  def get_MPP_LVK(self):
    Voc, Isc, Vmp, Imp = self.meas
    b, IL, I0, Rs, Gp = self.params
    return Eq(self.Ipv.subs({self.Ik:Imp, self.Vk:Vmp}), Imp)
      
  @property
  def get_dependence_IL_I0_Gp(self):
    """determination of the dependence of IL, I0 and Gp on Rs and b"""
    # unpacking variables
    b, IL, I0, Rs, Gp = self.params

    # General definition (get matrix form)
    A1, b1 = linear_eq_to_matrix([self.get_OC_LVK, 
                                  self.get_SC_LVK, 
                                  self.get_MPP_LVK], [IL, I0, Gp])
    return A1.solve(b1)
    

  @property
  def get_F0(self):
    # unpacking variables
    Vmp, Imp = self.meas[2:]
    Rs = self.params[3]
    
    # derived from the expression of the equivalent circuit of a PV array
    Ipv_dV = diff(self.Ik-self.Ipv, self.Vk)
    Ipv_dI = collect(diff(self.Ik-self.Ipv, self.Ik), Rs) 

    # replace dI_dV
    # => collet only positive values => use -dI_dV
    Ipv_dI = Ipv_dI.subs({Ipv_dV:self.dI_dV}) 

    # generic dI/dV from (13)
    generic_dI_dV = self.dI_dV + Ipv_dV*Ipv_dI

    # generic operating point from (12)
    generic_operating = Eq(self.dIV_dV - self.Ik - self.Vk*self.dI_dV, 0)

    # if dIV/dV = dP/dV = 0
    dI_dV_sol = solve(generic_operating.subs({self.dIV_dV:0}), self.dI_dV)[0] 
    F0 = simplify(generic_dI_dV.subs({self.dI_dV:dI_dV_sol})*self.Vk)
    F0 = F0.subs({self.Vk:Vmp, self.Ik:Imp})
    return F0




  def get_temperature_equations(self, model):
    """
    obtaining derivatives with respect to the
    temperature of the model

    definition of derivative
        dI/dT = partial_I(S,T) /partial_T 
            - Ik_term * partial_Ik /partial_T 
            + Vk_term * partial_Vk /partial_T
    """
    # unpacking variables
    # general params
    b, IL, I0, Rs, Gp = self.params

    # proposal model parameters (symbolic)
    if model.upper() == '11PFF':
      b_fun, IL_fun, I0_fun, Rs_fun, Gp_fun = self.modelSym11PFF.params(self.S, self.T)
    elif model.upper() == '7PFF':
      b_fun, IL_fun, I0_fun, Rs_fun, Gp_fun = self.modelSym7PFF.params(self.S, self.T)

    # Evaluate the expression equivalent circuit of a PV array using the proposed expressions 
    Ipv_sfun = self.Ipv.subs({b:b_fun, IL:IL_fun, I0:I0_fun, Rs:Rs_fun, Gp:Gp_fun})

    # ====================================================================
    #         derived from the expression of the equivalent  
    #         circuit of a PV array in terms of temperature
    # ====================================================================
    # derive the resulting expression with respect to temperature
    dIpv_dT1 = diff(Ipv_sfun, self.T)

    # Calculation of the second and third term of the derivative
    Ipv_dI = diff(self.Ik-Ipv_sfun, self.Ik)
    Ipv_dV = diff(self.Ik-Ipv_sfun, self.Vk)

    # concatenation of results 
    Ipv_dT = dIpv_dT1 - self.dIk_dT*Ipv_dI - self.dVk_dT*Ipv_dV 

    # ====================================================================
    #  derivative as a function of temperature of the expression 
    #     -> generic operating point (13):
    # ====================================================================
    # derived from the expression of the equivalent circuit of a PV array
    Ipv_dV2 = diff(self.Ik-Ipv_sfun, self.Vk)
    Ipv_dI2 = collect(diff(self.Ik-Ipv_sfun, self.Ik), Rs) 

    # replace dI_dV
    Ipv_dI2 = Ipv_dI2.subs({Ipv_dV2:self.dI_dV}) 

    # generic dI/dV from (13)
    generic_dI_dV = self.dI_dV + Ipv_dV2*Ipv_dI2

    # generic operating point from (12)
    generic_operating = Eq(self.dIV_dV - self.Ik - self.Vk*self.dI_dV, 0)

    # if dIV/dV = dP/dV = 0
    dI_dV_sol = solve(generic_operating.subs({self.dIV_dV:0}), self.dI_dV)[0] 
    F0 = simplify(generic_dI_dV.subs({self.dI_dV:dI_dV_sol})*self.Vk)

    # derivative in terms of temperature
    F0_dT1 = diff(F0, self.T)

    # Calculation of the second and third term of the derivative
    F0_dI = diff(F0, self.Ik)
    F0_dV = diff(F0, self.Vk)

    # concatenation of results 
    F0_dT = F0_dT1 + self.dIk_dT*F0_dI + self.dVk_dT*F0_dV 

    # temperature derivative for any point on curve IV    
    dIkVk_dT = self.dIk_dT*self.Vk + self.dVk_dT*self.Ik

    return [Ipv_dT, F0_dT, dIkVk_dT]


  def get_dIpv_sc(self, model):
    Voc, Isc, Vmp, Imp = self.meas
    alpha_spec, beta_spec, gamma_spec = self.spec
    [Ipv_dT, F0_dT, dIkVk_dT] = self.get_temperature_equations(model)
    # => replacing partial derivatives of Ik and Vk by sensitivity coefficients
    #    - dIsc_dT = alpha_spec
    return Ipv_dT.subs({self.Ik:Isc, self.Vk:0, self.dIk_dT:alpha_spec, self.dVk_dT:0})

  def get_dIpv_oc(self, model):
    Voc, Isc, Vmp, Imp = self.meas
    alpha_spec, beta_spec, gamma_spec = self.spec
    [Ipv_dT, F0_dT, dIkVk_dT] = self.get_temperature_equations(model)
    # => replacing partial derivatives of Ik and Vk by sensitivity coefficients
    #    - dVoc_dT = beta_spec
    return Ipv_dT.subs({self.Ik:0, self.Vk:Voc, self.dIk_dT:0, self.dVk_dT:beta_spec})

  def get_dIpv_mp(self, model):
    Voc, Isc, Vmp, Imp = self.meas
    [Ipv_dT, F0_dT, dIkVk_dT] = self.get_temperature_equations(model)
    # => replacing partial derivatives of Ik and Vk by sensitivity coefficients
    #    - dVmp_dT = gammaVmp
    #    - dImp_dT = gammaImp
    return Ipv_dT.subs({self.Ik:Imp, self.Vk:Vmp, self.dIk_dT:self.gammaImp, self.dVk_dT:self.gammaVmp})

  def get_dVpv_mp(self, model):
    Voc, Isc, Vmp, Imp = self.meas
    [Ipv_dT, F0_dT, dIkVk_dT] = self.get_temperature_equations(model)
    # => replacing partial derivatives of Ik and Vk by sensitivity coefficients
    #    - dVmp_dT = gammaVmp
    #    - dImp_dT = gammaImp
    return F0_dT.subs({self.Ik:Imp, self.Vk:Vmp, self.dIk_dT:self.gammaImp, self.dVk_dT:self.gammaVmp})

  def get_dPpv_mp(self, model):
    Voc, Isc, Vmp, Imp = self.meas
    [Ipv_dT, F0_dT, dIkVk_dT] = self.get_temperature_equations(model)
    # => replacing partial derivatives of Ik and Vk by sensitivity coefficients
    #    - dVmp_dT = gammaVmp
    #    - dImp_dT = gammaImp
    return dIkVk_dT.subs({self.Ik:Imp, self.Vk:Vmp, self.dIk_dT:self.gammaImp, self.dVk_dT:self.gammaVmp})
















    
    

















