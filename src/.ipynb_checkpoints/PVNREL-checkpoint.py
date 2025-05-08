import os, copy
import pandas as pd
import numpy as np
from scipy.constants import zero_Celsius as T0
from PVModel import Model11PFF, Model7PFF, Model6PFF, ModelA5PFF, Model5PFF

from scipy import integrate

source_path = os.getcwd()
results_path = os.path.join(source_path, 'NREL_results')
if not all([os.path.exists(results_path), os.path.isdir(results_path)]):
    os.makedirs(results_path)

class NREL_CECModel(object):
  @property
  def k11PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'mI0', 'mRs', 'mGp', 
            'alphaT', 'deltaI0', 'deltaRs', 
            'T_ref', 'S_ref']
  
  @property
  def k7PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT', # alphaIsc
            'mI0', 'deltaRs', 
            'T_ref', 'S_ref']
  
  @property
  def k6PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT', # alphaIsc
            'mIL',
            'T_ref', 'S_ref']
  
  @property
  def kA5PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT',# alphaIsc
            'Adjust',
            'T_ref', 'S_ref']


  @property
  def k5PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'alphaT',# alphaIsc
            'T_ref', 'S_ref']

  @property
  def kcolumns(self):
    return ['Isc', 'Pmp', 'Imp', 'Vmp', 'Voc']



  def datasheet_generator(self, df_summary, 
                        n_samples=30,
                        datasheet_config={
                                         'STC':{'S':1000, 'T':25},
                                        'NOCT':{'S': 800, 'T':45},
                                         'LIC':{'S': 200, 'T':25},
                                    }
                       ):
    assert isinstance(datasheet_config, dict)
    assert all(['STC' in datasheet_config.keys(), 
                'NOCT' in datasheet_config.keys(), 
                'LIC' in datasheet_config.keys()])
    assert all(['S' in datasheet_config['STC'].keys(), 
                'T' in datasheet_config['STC'].keys(), 
                'S' in datasheet_config['NOCT'].keys(), 
                'T' in datasheet_config['NOCT'].keys(), 
                'S' in datasheet_config['LIC'].keys(),
                'T' in datasheet_config['LIC'].keys()
               ])
    df_summary0 = copy.deepcopy(df_summary)

    
    def get_closest(dfk_data, S, T):
        df_closest = ((dfk_data[['S','T']]-(S, T+T0))/([S, T+T0])).apply(np.square).sum(axis=1)
        df_closest = df_closest.sort_values()
        dffk = pd.DataFrame(dfk_data.loc[df_closest.iloc[:n_samples].index].mean()).T
        dffk['T']-=T0
        dffk = dffk.rename(columns={'Vmp':'Vpmax', 'Imp':'Ipmax', 'Pmp':'Pmax'})
        dffk = dffk[['S', 'T', 'Voc', 'Isc', 'Vpmax', 'Ipmax', 'Pmax']]
        return dffk
        
    for tech, PVModule in df_summary.index:
        print(PVModule+' '*10, end="\r")
        dfk_data = self.gel_all_csv(PVModule)
        
        dfa = get_closest(dfk_data, datasheet_config['STC']['S'], datasheet_config['STC']['T'])
        dfa.columns = pd.MultiIndex.from_product([['DataSheet'], ['STC'], dfa.columns])
        
        dfb = get_closest(dfk_data, datasheet_config['NOCT']['S'], datasheet_config['NOCT']['T'])
        dfb.columns = pd.MultiIndex.from_product([['DataSheet'], ['NOCT'], dfb.columns])
        
        dfc = get_closest(dfk_data, datasheet_config['LIC']['S'], datasheet_config['LIC']['T'])
        dfc.columns = pd.MultiIndex.from_product([['DataSheet'], ['LIC'], dfc.columns])
        
        dff = pd.concat([dfa, dfb, dfc], axis=1)
        dff.index = pd.MultiIndex.from_tuples([(tech, PVModule)])
    
        df_summary0.loc[[(tech, PVModule)], dff.columns]  = dff
    print('end'+' '*10, end="\r")
    return df_summary0


    
  def __init__(self, NREL_path):
    self.Cocoa_path = os.path.join(NREL_path, 'Cocoa')
    self.Eugene_path = os.path.join(NREL_path, 'Eugene')
    self.Golden_path = os.path.join(NREL_path, 'Golden')
    self.LiteratureParams = pd.read_csv(os.path.join(NREL_path, 'ModelParams2023.csv'), index_col=[0, 1])
    self.LiteratureParams.T_ref+=T0
  


  def gel_all_csv(self, PVModule, withIV=False):
    def read_csv(path_csv):
        folder = path_csv.split(os.sep)[-2]
        if os.path.isfile(path_csv):
            dfa = pd.read_csv(path_csv, usecols=range(15), skiprows=2, sep=',', header=None, low_memory=False)
            dfa.columns = dfa.iloc[0]
            dfa = dfa.iloc[1:]
            dfa = dfa.set_index([dfa.columns[0]])
            dfa = dfa.astype(float)
            dfa = dfa.iloc[:,~dfa.columns.str.contains('uncertainty')]
            dfa.columns = dfa.columns.str.replace(r" \(.*\)", "", regex=True)
            dfa.index.name = 'Time'
            dfa = dfa.rename(columns={
                'POA irradiance CMP22 pyranometer':'S',
                'PV module back surface temperature':'T',
            })
            dfa['T']+=T0
            dfa = dfa.reset_index()
            dfa.insert(0, 'Place', folder)
            dfa = dfa.set_index(['Place', 'Time'])
            return dfa
    def read_csv_withIV(path_csv):
        folder = path_csv.split(os.sep)[-2]
        if os.path.isfile(path_csv):
            # dfa = pd.read_csv(path_csv, usecols=range(42), skiprows=2, sep=',', header=None, low_memory=False)
            dfa = pd.read_csv(path_csv, sep='\t', lineterminator='\r', skiprows=2, header=None, low_memory=False)
            dfa = dfa.iloc[:,0].str.strip().str.split(',', expand=True)
            dfa.iloc[0, dfa.iloc[[0]].isna().values[0]] = np.array(range(dfa.iloc[0].isna().sum())).astype(str)
            dfa.columns = dfa.iloc[0]
            dfa = dfa.iloc[1:]
            dfa = dfa.set_index([dfa.columns[0]])
            dfa = dfa.iloc[:,~dfa.columns.str.contains('uncertainty|FF|Change|Global|Diffuse|Direct|Daily|Precipitation|Relative|Atmospheric|Dry|MT5|soiling')]
            dfa.index.name = 'Time'
            dfa.columns = dfa.columns.str.replace(r" \(.*\)", "", regex=True)
            dfa = dfa.rename(columns={
                'POA irradiance CMP22 pyranometer':'S',
                'PV module back surface temperature':'T',
                'Number of I-V curve data pairs': 'Npts'
            })
            dfa = dfa.reset_index()
            dfa.insert(0, 'Place', folder)
            dfa = dfa.set_index(['Place', 'Time'])
            
            dfa = dfa.fillna(np.nan)
            dfa = dfa.astype(float)
            dfa['T']+=T0
            return dfa.iloc[:-1]
    all_csv = []
    all_csv+=[os.path.join(self.Cocoa_path, csv)  for csv in os.listdir(self.Cocoa_path) if PVModule in csv]
    all_csv+=[os.path.join(self.Eugene_path, csv) for csv in os.listdir(self.Eugene_path) if PVModule in csv]
    all_csv+=[os.path.join(self.Golden_path, csv) for csv in os.listdir(self.Golden_path) if PVModule in csv]
    if withIV:
        dfs = pd.concat([read_csv_withIV(path_csv) for path_csv in all_csv])
    else:
        dfs = pd.concat([read_csv(path_csv) for path_csv in all_csv])
    return dfs
  
  def get_model(self, df_iter, model):
    model = model.upper()
    PVModule = df_iter.name[1]
    df_iter = pd.DataFrame(df_iter).T
    df_params = df_iter.iloc[:,df_iter.columns.get_level_values(0).str.contains('Solution_{0}'.format(model))]
    df_params = df_params.T.droplevel(0).droplevel(1).T
    if model == '11PFF':
      return Model11PFF(*df_params[self.k11PFF].values.tolist()[0])
    else:
      df_params = self.LiteratureParams.loc[[(PVModule, model)]]
      if model == '7PFF':
        return Model7PFF(*df_params[self.k7PFF].values.tolist()[0])
      elif model == '6PFF':
        return Model6PFF(*df_params[self.k6PFF].values.tolist()[0])
      elif model == 'A5PFF':
        return ModelA5PFF(*df_params[self.kA5PFF].values.tolist()[0])
      elif model == '5PFF':
        return Model5PFF(*df_params[self.k5PFF].values.tolist()[0])



  def predict(self, PVModule, df_iter, model='11PFF', n_samples=1000):
    def task(df):
      [Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m] = mPFF.predict(df['S'], df['T'])
      df['Isc_m'] = Isc_m
      df['Pmp_m'] = Pmp_m
      df['Imp_m'] = Imp_m
      df['Vmp_m'] = Vmp_m
      df['Voc_m'] = Voc_m
      return df
    mPFF = self.get_model(df_iter, model=model)
    resultsExport_path = os.path.join(results_path, model, PVModule)
    if not all([os.path.exists(resultsExport_path), os.path.isdir(resultsExport_path)]):
      os.makedirs(resultsExport_path)
    csvExport_path = os.path.join(resultsExport_path, '{0}.csv')
    dfs0 = self.gel_all_csv(PVModule)
    k = 0
    while True:
      df = dfs0.iloc[n_samples*k:n_samples*(k+1)]
      if not os.path.exists(csvExport_path.format(k)):
        df_predict = df.apply(task, axis=1)
        df_predict.to_csv(csvExport_path.format(k))
      if df.shape[0]==0:
        break
      k+=1

    lst_predict = []
    for csvFile in os.listdir(resultsExport_path):
      if os.path.isfile(os.path.join(resultsExport_path, csvFile)):
        dff = pd.read_csv(os.path.join(resultsExport_path, csvFile), index_col=[0,1], parse_dates=[1])
        if dff.shape[0]:
          lst_predict.append(dff)
    df_predict = pd.concat(lst_predict)
    
    filterPredict = df_predict.columns.str.contains('_m')
    df_meas = df_predict.iloc[:,~filterPredict]
    df_pred = df_predict.iloc[:,filterPredict]
    df_pred.columns = pd.MultiIndex.from_product([[f'Prediction_{model}'], df_pred.columns.str.rstrip('_m')])
    df_meas.columns = pd.MultiIndex.from_product([['Measurement'], df_meas.columns])
    df_predict = pd.concat([df_meas, df_pred], axis=1)
    return df_predict

      

  def error(self, dfp):
    cPred = dfp.columns.get_level_values(0).str.contains('Prediction')
    cMeas = dfp.columns.get_level_values(0).str.contains('Measurement')
    cVars = dfp.columns.get_level_values(1).str.contains('|'.join(self.kcolumns))
    dfe  = (1-dfp.iloc[:, cPred]/dfp.iloc[:, cMeas&cVars].values).abs()*100
    lvl0 = dfe.columns.get_level_values(0)
    lvl1 = dfe.columns.get_level_values(1)
    lvl0 = lvl0.str.replace('Prediction', 'Error')
    dfe.columns = pd.MultiIndex.from_tuples(tuple(zip(lvl0, lvl1)))
    dfp = pd.concat([dfp, dfe], axis=1)
    return dfp











  def get_summary(self, dfp, applyFilter=False, STfilter=None):
    def task1(df, name):
      dfx = pd.DataFrame(df.mean()).T
      dfx.insert(0, ('pts',''), df.shape[0])
      return dfx
    
    def task2(df, name):
        dfq = df.quantile([0.05, 0.95])
        dict1 = {}
        for col in dfq:
            dff = df.loc[df[col].between(*dfq[col]), col]
            dict1[col] = dff.mean()
        dfx = pd.DataFrame(dict1.values(), index=pd.MultiIndex.from_tuples(dict1.keys())).T    
        dfx.insert(0, ('pts',''), df.shape[0])
        return dfx
    
    
    cErro = dfp.columns.get_level_values(0).str.contains('Error')
    if all([isinstance(STfilter, dict), applyFilter]):
      filterS = dfp.iloc[:,0].between(STfilter['S'][0], STfilter['S'][1])
      filterT = dfp.iloc[:,1].between(STfilter['T'][0]+T0, STfilter['T'][1]+T0)
      dfp = dfp.loc[filterS&filterT]
    
    dfe_with = dfp.iloc[:,cErro].groupby(level=[0, 1]).apply(lambda df: task1(df, df.name)).droplevel(2)
    dfe_without = dfp.iloc[:,cErro].groupby(level=[0, 1]).apply(lambda df: task2(df, df.name)).droplevel(2)
    return [dfe_with, dfe_without]




  def get_pts_results(self, dfp):
    ST_STC = {'S':[980, 1020], 'T':[22, 28]}
    ST_NOCT = {'S':[780, 820], 'T':[47, 53]}
    ST_LIC = {'S':[180, 220], 'T':[22, 28]}
    
    dfe_with, dfe_without = self.get_summary(dfp, applyFilter=True, STfilter=ST_STC)
    cVars0  = dfe_without.columns.get_level_values(0).str.contains('pts')
    cVars1  = dfe_without.columns.get_level_values(1).str.contains('Pmp')
    df_STC = dfe_without.iloc[:, cVars0|cVars1].droplevel(level=1, axis=1)
    
    dfe_with, dfe_without = self.get_summary(dfp, True, ST_NOCT)
    cVars0  = dfe_without.columns.get_level_values(0).str.contains('pts')
    cVars1  = dfe_without.columns.get_level_values(1).str.contains('Pmp')
    df_NOCT = dfe_without.iloc[:, cVars0|cVars1].droplevel(level=1, axis=1)
    
    dfe_with, dfe_without = self.get_summary(dfp, True, ST_LIC)
    cVars0  = dfe_without.columns.get_level_values(0).str.contains('pts')
    cVars1  = dfe_without.columns.get_level_values(1).str.contains('Pmp')
    df_LIC = dfe_without.iloc[:, cVars0|cVars1].droplevel(level=1, axis=1)
    
    index = df_STC.index
    index = index.union(df_NOCT.index)
    index = index.union(df_LIC.index)
    df = []
    for idx in index:
      try:
        n1 = df_STC.loc[idx].iloc[0]
        s1 = df_STC.loc[idx].iloc[1:]
      except:
        n1 = 0
        s1 = 0
    
      try:
        n2 = df_NOCT.loc[idx].iloc[0]
        s2 = df_NOCT.loc[idx].iloc[1:]
      except:
        n2 = 0
        s2 = 0
    
      try:
        n3 = df_LIC.loc[idx].iloc[0]
        s3 = df_LIC.loc[idx].iloc[1:]
      except:
        n3 = 0
        s3 = 0
      dfk = pd.DataFrame(s1+s2+s3).T/3
      dfk.insert(0, 'pts', n1+n2+n3)
      df.append(dfk)
    
    df = pd.concat(df)
    df.columns = pd.MultiIndex.from_product([['STC+NOCT+LIC'], df.columns.str.lstrip('Error_')])
    return df

  def get_lattice_results(self, dfp):
    STfilter = {'S':[200, 1100], 'T':[15, 75]}
    dfe_with, dfe_without = self.get_summary(dfp, True, STfilter)
    cVars0  = dfe_without.columns.get_level_values(0).str.contains('pts')
    cVars1  = dfe_without.columns.get_level_values(1).str.contains('Pmp')
    df = dfe_without.iloc[:, cVars0|cVars1].droplevel(level=1, axis=1)
    df.columns = pd.MultiIndex.from_product([['Lattice'], df.columns.str.lstrip('Error_')])
    return df

  def get_alldata_results(self, dfp):
    dfe_with, dfe_without = self.get_summary(dfp, False)
    cVars0  = dfe_without.columns.get_level_values(0).str.contains('pts')
    cVars1  = dfe_without.columns.get_level_values(1).str.contains('Pmp')
    df = dfe_without.iloc[:, cVars0|cVars1].droplevel(level=1, axis=1)
    df.columns = pd.MultiIndex.from_product([['All'], df.columns.str.lstrip('Error_')])
    return df








    
  def get_energy(self, df_pred): 
    def task(df, key):
        df = df.sort_index(level=[1, 2])
        time_x = df.index.get_level_values(2)
        dx = time_x[1]-time_x[0]
        dfy = df.apply(lambda y: integrate.trapezoid(y=y, x=time_x, axis=0).astype('timedelta64[h]').astype(float))
        dfy = dfy.droplevel(1)
        dfy = pd.DataFrame(dfy).T/1e3 # to kW
        columns = pd.MultiIndex.from_product([['Integral'],dfy.columns.str.replace(r'Prediction_', '')])
        dfy.columns = columns
        dfy.insert(0,'pts', df.shape[0])
        return dfy 
    dfx0 = df_pred.groupby(level=[0, 1]).apply(lambda x: task(x, x.name)).droplevel(2)
    dfx1 = np.abs(dfx0.iloc[:, 2:] - dfx0.iloc[:, [1]].values)
    dfx1.columns = pd.MultiIndex.from_product([['Difference'], dfx1.columns.get_level_values(1)])
    dfx = pd.concat([dfx0, dfx1], axis=1)
    return dfx



  def __call__(self, models, df_summary):
    list_predict = []
    for nx, ((tech, PVModule), df_iter) in enumerate(df_summary.iterrows()):
        print(PVModule)
        dff = []
        for model in models:
            print('-> {:<20s}'.format(model), end='\r')
            dfp = self.predict(PVModule, df_iter, model)
            dfp = self.error(dfp)
            dfp.index = pd.MultiIndex.from_tuples([(PVModule, idx[0], idx[1]) for idx in dfp.index])
            dff.append(dfp)
        dff = pd.concat(dff, axis=1)
        dff = dff.loc[:, ~dff.columns.duplicated()]
        list_predict.append(dff)            

    dfp = pd.concat(list_predict)
    df_STC_NOCT_LIC = self.get_pts_results(dfp)
    df_lattice = self.get_lattice_results(dfp)
    df_ALL = self.get_alldata_results(dfp)
    dfs = pd.concat([df_STC_NOCT_LIC, df_lattice, df_ALL], axis=1).round(2)

    cPred = dfp.columns.get_level_values(0).str.contains('Prediction|Measurement')
    cVars  = dfp.columns.get_level_values(1).str.contains('Pmp')
    df_energy = self.get_energy(dfp.iloc[:, cVars&cPred]).round(2)
    return dfp, dfs, df_energy












