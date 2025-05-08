import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec


def replace_unnamed(level):
  return tuple('' if 'Unnamed' in str(col) else col for col in level)


def CEC2CSV(CECpath):
  """
    CECpath='BBDD/PV_Module_List_Full_Data_ADA.xlsx'
    df = CEC2CSV(CECpath)
  """
  #=========================================
  # read xlsx
  # ==========================================
  df = pd.read_excel(CECpath, engine='openpyxl', skiprows=16, header=[0,1])
  columns = []
  for k in df.keys():
    if "Unnamed" not in k[1]:
      columns.append(' '.join(k).replace('\n',' '))
    else:
      columns.append(k[0].replace('\n',' '))
  df.columns = columns
  df.insert(0, 'Model', df.iloc[:,:2].apply(lambda x: "_".join(x), axis =1))

  columns = ['S', 'T', 'Voc', 'Isc','Vpmax', 'Ipmax', 'Pmax']
  df1 = copy.deepcopy(df)
  df1 = df1.drop(columns=df.columns[[1,2,3,4,6,7,8,9,10,12,15,24,25,-8,-7,-6,-5,-4,-3,-2,-1]])
  df1 = df1.drop_duplicates()
  df1 = df1.rename(columns={
      'Model':'Module',
      'αIsc (%/°C)':'alpha',
      'βVoc (%/°C)':'beta',
      'γPmax (%/°C)':'gamma',
      })

  df1 = df1.set_index(['Technology', 'Module'])
  df1 = df1.sort_index()
  dfx = df1[['alpha', 'beta', 'gamma', 'N_s', 'N_p']]

  df_STC = df1.iloc[:, df1.columns.str.contains('Nameplate')]
  df_STC.columns = df_STC.columns.str.replace('Nameplate ', '')
  df_STC.columns = df_STC.columns.str.replace(r" \(.*\)", "", regex=True)
  df_STC.insert(0, 'S', 1000)
  df_STC.insert(1, 'T', 25)
  df_STC = df_STC[columns]

  df0 = copy.deepcopy(dfx)
  df0.loc[df1.index, 'alpha'] = df0.loc[df1.index, 'alpha']*df_STC.loc[df1.index, 'Isc']/100
  df0.loc[df1.index, 'beta']  = df0.loc[df1.index,  'beta']*df_STC.loc[df1.index, 'Voc']/100
  df0.loc[df1.index, 'gamma'] = df0.loc[df1.index, 'gamma']*df_STC.loc[df1.index, 'Pmax']/100

  df_NOCT = df1.iloc[:, df1.columns.str.contains('NOCT')]
  df_NOCT.columns = df_NOCT.columns.str.replace(', NOCT', '').str.replace(' NOCT', '')
  df_NOCT.columns = df_NOCT.columns.str.replace(r" \(.*\)", "", regex=True)
  df_NOCT.columns = df_NOCT.columns.str.replace('P', 'p')
  df_NOCT = df_NOCT.rename(columns={'Average':'T'})
  df_NOCT.insert(0, 'S', 800)
  df_NOCT.insert(1, 'Pmax', df_NOCT.Vpmax*df_NOCT.Ipmax)
  df_NOCT.insert(2, 'Voc', np.nan)
  df_NOCT.insert(3, 'Isc', np.nan)
  df_NOCT = df_NOCT[columns]

  df_LIC = df1.iloc[:, df1.columns.str.contains('low')]
  df_LIC.columns = df_LIC.columns.str.replace(', low', '')
  df_LIC.columns = df_LIC.columns.str.replace(r" \(.*\)", "", regex=True)
  df_LIC.columns = df_LIC.columns.str.replace('P', 'p')
  df_LIC.insert(0, 'S', 200)
  df_LIC.insert(0, 'T', 25)
  df_LIC.insert(2, 'Pmax', df_LIC.Vpmax*df_LIC.Ipmax)
  df_LIC.insert(3, 'Voc', np.nan)
  df_LIC.insert(4, 'Isc', np.nan)
  df_LIC = df_LIC[columns]

  df_STC.columns  = pd.MultiIndex.from_product([['STC'], df_STC.columns])
  df_NOCT.columns = pd.MultiIndex.from_product([['NOCT'], df_NOCT.columns])
  df_LIC.columns  = pd.MultiIndex.from_product([['LIC'], df_LIC.columns])
  df2 = pd.concat([df_STC, df_NOCT,df_LIC], axis=1)
  for col in df0.columns:
    df2.insert(df2.shape[1], col, df0[col])
  df2.columns = pd.MultiIndex.from_tuples([tuple(['DataSheet']+list(k)) for k in df2.columns])

  df2 = df2.reset_index()
  df2.Module = df2.Module.str.replace('/','_')
  df2 = df2.set_index(['Technology', 'Module'])
  df2 = df2.sort_index()
  return df2


def SummaryByTech(df_summary, tol= 0.01, to_latex=False):
    # filters
    cDS  = df_summary.columns.get_level_values(0).str.contains('DataSheet')
    c11  = df_summary.columns.get_level_values(0).str.contains('Prediction_11PFF')
    cPmp = df_summary.columns.get_level_values(2).str.contains('Pmp|Pmax')
    df_tol = (1-df_summary.iloc[:, c11&cPmp]/df_summary.iloc[:, cDS&cPmp].values).apply(np.square).sum(axis=1)/3

    # apply tolerance
    df_tol[~df_summary.iloc[:, 26].astype(bool)] = np.nan
    df_tol[~df_summary.iloc[:, 155].astype(bool)] = np.nan

    if to_latex:
        print("{:<15s} {:>7s}|{:>7s}|{:>7s}".format('name','n', 'y/y', 'y/n'))
        print('\hline')
        for nx, (key, dfk) in enumerate(df_tol.groupby(level=0)):
          if nx< len(df_tol.groupby(level=0))-1:
            print("{:<15s}&{:>7d}&{:>7d}&{:>7d}\\\\".format(
                key, dfk.isna().sum(), (dfk[~dfk.isna()]<=tol).sum(), dfk.shape[0]-(dfk[~dfk.isna()]<=tol).sum()  ))
          else:
            print("{:<15s}&{:>7d}&{:>7d}&{:>7d}\B\\\\".format(
                key, dfk.isna().sum(), (dfk[~dfk.isna()]<=tol).sum(), dfk.shape[0]-(dfk[~dfk.isna()]<=tol).sum()  ))
        print('\hline')
        print("{:<15s}&{:>7d}&{:>7d}&{:>7d}\\\\".format('Total' ,df_tol.isna().sum(), (df_tol<=tol).sum(), df_tol.shape[0]-(df_tol<=tol).sum()  ))
        print('\hline')
    else:
        print("{:<15s} {:>7s}|{:>7s}|{:>7s}".format('name','n', 'y/y', 'y/n'))
        print('-'*40)
        for key, dfk in df_tol.groupby(level=0):
            isna = dfk.isna().sum()
            dfk = dfk[~dfk.isna()]
            print("{:<15s} {:>7d}|{:>7d}|{:>7d}".format(
                key, 
                isna, 
                (dfk<=tol).sum(), 
                (dfk>tol).sum(), 
            ))
        print('-'*40)
        print("{:<15s} {:>7d}|{:>7d}|{:>7d}".format(
            'Total', 
            df_tol.isna().sum(), 
            (df_tol[~df_tol.isna()]<=tol).sum(), 
            (df_tol[~df_tol.isna()]>tol).sum(),
        ))

def tolatex(df):
    return print(df.to_latex(float_format="{:0.2f}".format))





def summary(df_summary, tol=5):
    print("{:14s}  {:>8s} | {:>8s} | {:>8s} | {:>8s}".format('Technology', 'No', '<=eta', '>eta', 'Total' ))
    print('-'*60)
    f_DS    = df_summary.columns.get_level_values(0).str.contains('DataSheet')
    f_conv  = df_summary.columns.get_level_values(1).str.contains('converge')
    f_7pff  = df_summary.columns.get_level_values(0).str.contains('_7PFF')
    f_11pff = df_summary.columns.get_level_values(0).str.contains('_11PFF')
    f_pmp   = df_summary.columns.get_level_values(2).str.contains('Pmp|Pmax')
    f_econv = df_summary.columns.get_level_values(1).str.contains('eF')
    
    dff0 = df_summary.iloc[:, f_pmp&f_DS].droplevel(level=[0,2], axis=1)
    dff1 = df_summary.iloc[:, f_pmp&f_11pff].droplevel(level=[0,2], axis=1)
    dff2 = df_summary.iloc[:, f_11pff&f_conv].droplevel(level=[0,2], axis=1)
    
    df_eta  = abs(1-dff1/dff0).mean(axis=1)*100
    df_loss = df_summary.iloc[:,f_econv].apply(np.square).sum(axis=1)
    
    f_eta = df_eta.iloc[dff2.values]<=tol
    
    all_values = []
    for k, dfk in df_summary.groupby(level=0):
        try:
            n_pv    = dfk.shape[0]
            f0_conv  = dfk.columns.get_level_values(1).str.contains('converge')
            f0_conv  = dfk.iloc[:, f_11pff&f0_conv]
            ff_eta  = f_eta.loc[f0_conv.iloc[f0_conv.values].index]
            n_conv  = ff_eta.sum()
            n_convr = (~ff_eta).sum()
            n_nconv = n_pv - ff_eta.shape[0]
            print("{:14s}  {:>8d} | {:>8d} | {:>8d} | {:>8d}".format(k, n_nconv, n_conv, n_convr, n_pv ))
            all_values.append([n_nconv, n_conv, n_convr, n_pv])
        except Exception as e:
            pass
        
        
    print('-'*60)
    print("{:14s}  {:>8d} | {:>8d} | {:>8d} | {:>8d}".format('total', *np.array(all_values).sum(axis=0)))

    f0_conv = df_summary.columns.get_level_values(1).str.contains('converge')
    f0_conv = df_summary.iloc[:, f_11pff&f0_conv]
    
    df_nc = df_summary[~f0_conv.values]
    df_c1 = df_summary[f0_conv.values][f_eta.values]
    df_c2 = df_summary[f0_conv.values][~f_eta.values]
    

    return df_nc, df_c1, df_c2








from matplotlib.ticker import FixedLocator, MaxNLocator, LinearLocator, AutoLocator
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(
  {
    'font.size': 17,
    "text.usetex": False,
    "font.family": "serif",
    "font.sans-serif": ['Computer Modern'],
  }
)


def plot_histogram(dff, emin=0, emax=40, cumulative=False):
    f_DS    = dff.columns.get_level_values(0).str.contains('DataSheet')
    f_conv  = dff.columns.get_level_values(1).str.contains('converge')
    f_7pff  = dff.columns.get_level_values(0).str.contains('_7PFF')
    f_11pff = dff.columns.get_level_values(0).str.contains('_11PFF')
    f_pmp   = dff.columns.get_level_values(2).str.contains('Pmp|Pmax')
    f_econv = dff.columns.get_level_values(1).str.contains('eF')

    df1 = dff.iloc[:, f_DS&f_pmp].droplevel(level=[0,2], axis=1) 
    df2 = dff.iloc[:, f_11pff&f_pmp].droplevel(level=[0,2], axis=1)
    dfe = abs(1-df2/df1).mean(axis=1)*100
    
    fig = plt.figure(tight_layout=True, figsize=(10, 6))
    gs = gridspec.GridSpec(2, 6)
    axs = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    
    
    dffo = []
    dffe = []
    print(48*'-')
    for idk, (k, dfk) in enumerate(dfe.groupby(level=0)):
        dfo = dfk[~dfk.between(*[0, emax])]
        dfk = dfk[dfk.between(*[0, emax])]
        dffo.append(dfo)
        dffe.append(dfk)
    
        print('{:12s} | {:7.3f} | {:7.3f} ({:7.3f} - {:7.3f}) | {:7.3f}'.format(k, dfk.min(), abs(dfk).mean(), abs(dfk).var(), abs(dfk).std(), dfk.max()))
        
        if k == 'CdTe':
            ax = axs[0]
        elif k == 'CIGS':
            ax = axs[1]
        else:
            ax = axs[idk]
        sns.histplot(dfk.values, alpha=0.5, bins=15, stat='percent', ax=ax, cumulative=cumulative)
        ax.axvline(dfk.values.mean(), c='r', ls='--', lw=1)
        ax.set_title(k)
        
        ax.set_xlabel('MAPE (%)')
        ax.set_ylabel('Percent (%)')
        ax.xaxis.set_major_locator(MaxNLocator(3))
        if cumulative:
            ax.set_ylim([0, 100])
            ax.yaxis.set_major_locator(MaxNLocator(4))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(3))
    
    dfo = pd.concat(dffo)
    dfe = pd.concat(dffe)
    print(48*'-')
    print('{:12s} | {:7.3f} | {:7.3f} ({:7.3f} - {:7.3f}) | {:7.3f}'.format('Total', dfe.min(), abs(dfe).mean(), abs(dfe).var(), abs(dfe).std(), dfe.max()))
    print(48*'-')
    print(dfe.shape[0], ' | ', dfo.shape[0], ' | ', dff.shape[0])
    if cumulative:
        plt.savefig("FIgs/histogram_cum.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.savefig("FIgs/histogram.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    return dfe, dfo
