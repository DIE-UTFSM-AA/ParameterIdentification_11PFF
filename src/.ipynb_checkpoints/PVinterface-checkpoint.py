import os, sys, json, time
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from PVModel import Model11PFF, Model7PFF, Model6PFF, Model5PFF
from scipy.constants import zero_Celsius as T0


source_path = os.getcwd()
src_path = os.path.join(source_path, 'src')
assert all([os.path.exists(src_path), os.path.isdir(src_path)])
if src_path not in sys.path:
  sys.path.append(src_path)

class MainWindow(QDialog):
  def __init__(self):
    self.df_summary = pd.read_csv(os.path.join(src_path, 'ui', 'summary.csv'), index_col=[0,1], header=[0,1,2])
    QDialog.__init__(self)
    uic.loadUi(os.path.join(src_path, 'ui', 'main.ui'), self)
    self.SummaryOptions.setFixedWidth(200)
    self.PVPlotOptions.setFixedWidth(200)

    # ===========================================
    # get converge list of PVModules 
    # ===========================================
    def task(df):
      return df['Module'].values.tolist()
    filter_7PFF     = self.df_summary.columns.get_level_values(0).str.contains('7PFF')
    filter_11PFF    = self.df_summary.columns.get_level_values(0).str.contains('11PFF')
    filter_converge = self.df_summary.columns.get_level_values(1).str.contains('converge')
    
    df_7PFFc = self.df_summary.iloc[:, filter_7PFF&filter_converge]
    df_7PFFc = df_7PFFc.T.droplevel(0).droplevel(1).T
    df_7PFFc = df_7PFFc.loc[df_7PFFc.iloc[:,0].astype(bool)].drop(columns=df_7PFFc.columns)
    df_7PFFc = df_7PFFc.reset_index().groupby('Technology').apply(task)

    df_11PFFc = self.df_summary.iloc[:, filter_11PFF&filter_converge]
    df_11PFFc = df_11PFFc.T.droplevel(0).droplevel(1).T
    df_11PFFc = df_11PFFc.loc[df_11PFFc.iloc[:,0].astype(bool)].drop(columns=df_11PFFc.columns)
    df_11PFFc = df_11PFFc.reset_index().groupby('Technology').apply(task)

    self.all_dirs = dict()
    self.all_dirs['7PFF'] = df_7PFFc.to_dict()
    self.all_dirs['11PFF'] = df_11PFFc.to_dict()


    # ===========================================
    # calcule error of predictions 
    # ===========================================
    dfmeas = self.df_summary.iloc[:,self.df_summary.columns.get_level_values(0).str.contains('DataSheet')]
    dfmeas = dfmeas.T.droplevel(0).T
    dfmeas = dfmeas.iloc[:, dfmeas.columns.get_level_values(1).str.contains('Voc|Isc|Vpmax|Ipmax|Pmax')]

    def error(pto, model, 
              columns_meas = ['Vpmax', 'Ipmax', 'Pmax'], 
              columns_predict = ['Vmp', 'Imp', 'Pmp']):
      
      dfmeas_pto = dfmeas.iloc[: ,dfmeas.columns.get_level_values(0).str.contains(pto)]
      dfmeas_pto = dfmeas_pto.T.droplevel(0).T
      dfmeas_pto = dfmeas_pto[columns_meas]
      dfmeas_pto = dfmeas_pto.rename(columns=dict(zip(columns_meas,columns_predict)))
    
      dfmodel_pto = self.df_summary.iloc[:,self.df_summary.columns.get_level_values(0).str.contains('Prediction_{0}'.format(model))]
      dfmodel_pto = dfmodel_pto.T.droplevel(0).T
      dfmodel_pto = dfmodel_pto.iloc[:, dfmodel_pto.columns.get_level_values(0).str.contains(pto)]   
      dfmodel_pto = dfmodel_pto.T.droplevel(0).T
      dfmodel_pto = dfmodel_pto[columns_predict] 
      
      error_pto = (dfmeas_pto - dfmodel_pto)/dfmeas_pto *100
      error_pto.columns = pd.MultiIndex.from_product([['Error_{0}'.format(model)], [pto], error_pto.columns])
      return error_pto

    error_11PFF_LIC  = error('LIC',  '11PFF')
    error_11PFF_NOCT = error('NOCT', '11PFF')
    error_11PFF_STC  = error('STC',  '11PFF')
    error_11PFF = pd.concat([error_11PFF_LIC, error_11PFF_NOCT, error_11PFF_STC], axis=1)

    error_7PFF_LIC  = error('LIC',  '7PFF')
    error_7PFF_NOCT = error('NOCT', '7PFF')
    error_7PFF_STC  = error('STC',  '7PFF')
    error_7PFF = pd.concat([error_7PFF_LIC, error_7PFF_NOCT, error_7PFF_STC], axis=1)
    self.df_error = pd.concat([error_7PFF, error_11PFF], axis=1)
    
    # ===========================================
    # Initial configuration of objects
    # ===========================================
    # - Summary Tab
    self.selecDataHist.addItems([
        "dP/dV in STC", "dP/dV in NOCT", "dP/dV in LIC", 
        'LVK_MPP in NOCT', 'LVK_MPP in LIC',
        "eVmp", "eImp", "ePmp"
        ])

    self.selecModule_PVPlot.addItems(
        self.all_dirs[self.selecModel_PVPlot.currentText()][self.selecTech_PVPlot.currentText()]
        )
    self.selecAddDSP_PVPlot.setChecked(True)  
    
    # ===========================================
    # linking functions with graphic actions
    # ===========================================
    # - Summary Tab
    self.selecModelHist.currentIndexChanged.connect(self.ComboxModelHistEvent)    
    self.ButtonUpdateHist.clicked.connect(self.updateGraph)

    # - PVPlot Tab
    self.selecTech_PVPlot.currentIndexChanged.connect(self.ComboxTechnologyAndModelPVPlotEvent)
    self.selecModel_PVPlot.currentIndexChanged.connect(self.ComboxTechnologyAndModelPVPlotEvent)
    self.selecModule_PVPlot.currentIndexChanged.connect(self.compile)
    self.ButtonUpdate_PVPlot.clicked.connect(self.updateGraph)


    # ===========================================
    # initial render
    # ===========================================
    self.compile()
    self.updateGraph()
  
  def compile(self):
    current_Technology = self.selecTech_PVPlot.currentText() 
    current_PVModule = self.selecModule_PVPlot.currentText()

    df_summary = self.df_summary.loc[[(current_Technology, current_PVModule)],:]    
    df_Data = df_summary.iloc[:,df_summary.columns.get_level_values(0).str.contains('DataSheet')]
    df_Data = df_Data.T.droplevel(0).T

    # get first point (reference)
    self.meas_ds1 = df_Data.iloc[:,df_Data.columns.get_level_values(0).str.contains('STC')].values[0].tolist()
    self.meas_ds1[1] +=T0

    # get second point (complementary)
    self.meas_ds2 = df_Data.iloc[:,df_Data.columns.get_level_values(0).str.contains('NOCT')].values[0].tolist()
    self.meas_ds2[1] +=T0

    # get third point (complementary or regulatory)
    self.meas_ds3 = df_Data.iloc[:,df_Data.columns.get_level_values(0).str.contains('LIC')].values[0].tolist()
    self.meas_ds3[1] +=T0

    # other params
    self.alpha_spec, self.beta_spec, self.gamma_spec, self.Ns, self.Np = df_Data.iloc[:,-5:].T.droplevel(1).T.values[0]

    # get params of model
    self.df_params = df_summary.iloc[:,df_summary.columns.get_level_values(0).str.contains('Solution')]
  
    # get temperature behavior
    self.df_temperature = df_summary.iloc[:,df_summary.columns.get_level_values(0).str.contains('Temperature')]

  def ComboxModelHistEvent(self):
    print('active ComboxModelHistEvent')
    current_model = self.selecModelHist.currentText() 

    # reset children combox
    self.selecDataHist.clear()
    
    if current_model=='11PFF':
      self.selecDataHist.addItems([
        "dP/dV in STC", "dP/dV in NOCT", "dP/dV in LIC", 
        'LVK_MPP in NOCT', 'LVK_MPP in LIC',
        "eVmp", "eImp", "ePmp"
        ])
    elif current_model=='7PFF':
      self.selecDataHist.addItems([
          "dP/dV in STC", 
          'deltaI0', 'alphaIsc',
          "eVmp", "eImp", "ePmp"
          ])

  def ComboxTechnologyAndModelPVPlotEvent(self):
    print('active ComboxTechnologyPVPlotEvent')
    current_Technology = self.selecTech_PVPlot.currentText() 
    current_Model = self.selecModel_PVPlot.currentText()

    # reset children combox
    self.selecModule_PVPlot.currentIndexChanged.disconnect(self.compile)
    self.selecModule_PVPlot.clear()
    self.selecModule_PVPlot.currentIndexChanged.connect(self.compile)
    self.selecModule_PVPlot.addItems(self.all_dirs[current_Model][current_Technology])
    
  def updateGraph(self):
    plt.close('all')
    def ResetContent(Layout_object):
      # Remove all widgets from the QVBoxLayout
      while Layout_object.count() > 0:
        item = Layout_object.takeAt(0)
        if item.widget():
            item.widget().deleteLater()

    if self.tabWidget.currentIndex()==0:
      ResetContent(self.LayoutSummary)
      canvas = self.plot_histogram()
      self.LayoutSummary.addWidget(canvas)
    else:
      ResetContent(self.LayoutPVPlot)
      canvas = self.plot_PVCurve()
      self.LayoutPVPlot.addWidget(canvas)        

  def plot_PVCurve(self):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
      {
        'font.size': 10,
        "text.usetex": False,
        "font.family": "serif",
        "font.sans-serif": ['Computer Modern'],

        "axes.grid" : True, 
        "grid.color": "0.95",
        "grid.linestyle": "--",
      }
    )
    def tick_format(x, y):
      return "%03.2f"%x
    

    pts = 100

    # =========== model definition 
    current_Model = self.selecModel_PVPlot.currentText()     
    df_params = self.df_params.iloc[:, self.df_params.columns.get_level_values(0).str.contains(current_Model)]
    df_params = df_params.T.droplevel(0).droplevel(1).T

    if current_Model=='11PFF':
      mPFF = Model11PFF(*df_params[self.k11PFF].values.tolist()[0])
    elif current_Model=='7PFF':
      mPFF = Model7PFF(*df_params[self.k7PFF].values.tolist()[0])

    # figure
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(3, 6, height_ratios=(1, 9, 9))
    
    # electrical
    ax1 = fig.add_subplot(gs[1, 0:3])
    ax1.set_xlabel('$v_{pv}$ (V)')
    ax1.set_ylabel('$p_{pv}$ (W)')
    ax1.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax1.yaxis.set_major_locator(ticker.LinearLocator(6))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(tick_format))

    ax2 = fig.add_subplot(gs[1, 3:], sharex = ax1)
    ax2.set_xlabel('$v_{pv}$ (V)')
    ax2.set_ylabel('$i_{pv}$ (A)')
    ax2.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax2.yaxis.set_major_locator(ticker.LinearLocator(6))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(tick_format))


    # Temperature
    ax3 = fig.add_subplot(gs[2, 0:2])
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel(r'$\partial I/\partial T$ (A/°C)') 
    ax3.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    ax4 = fig.add_subplot(gs[2, 2:4], sharex = ax3)
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel(r'$\partial V/\partial T$ (V/°C)') 
    ax4.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax4.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    ax5 = fig.add_subplot(gs[2, 4:6], sharex = ax3)
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel(r'$\partial P/\partial T$ (W/°C)') 
    ax5.xaxis.set_major_locator(ticker.LinearLocator(4))
    ax5.xaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax5.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    # legend
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')    
    

    # =========== Reference conditions
    S, T, Voc, Isc, Vmp, Imp, Pmp = self.meas_ds1
    [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(S, T, pts)
    p1 = ax1.plot(V_model, P_model, label='Standar Test Condition\n{:5.2f}(W/m$^2$)\n{:5.2f}(°C)'.format(S, T-T0))
    color = p1[0].get_color()

    ax1.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color=color, marker='.', lw=0, label='Model')
    if self.selecAddDSP_PVPlot.isChecked():
      ax1.plot([0, Vmp, Voc], [0, Pmp, 0], color=color, marker='x', lw=0, label='DataSheet')

    ax2.plot(V_model, I_model, color=color)
    ax2.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color=color, marker='.', lw=0, label='Model')
    if self.selecAddDSP_PVPlot.isChecked():
      ax2.plot([0, Vmp, Voc], [Isc, Imp, 0], color=color, marker='x', lw=0, label='DataSheet')
    
    # =========== Complementary conditions (NOCT)
    S, T, Voc, Isc, Vmp, Imp, Pmp = self.meas_ds2
    [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(S, T, pts)
    
    p2 = ax1.plot(V_model, P_model, label='Nominal Operating Cell Temperature\n{:5.2f} (W/m$^2$)\n{:5.2f} (°C)'.format(S, T-T0))
    color = p2[0].get_color()

    ax1.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color=color, marker='.', lw=0, label='Model')
    if self.selecAddDSP_PVPlot.isChecked():
      ax1.plot([0, Vmp, Voc], [0, Pmp, 0], color=color, marker='x', lw=0, label='DataSheet')

    ax2.plot(V_model, I_model, color=color)
    ax2.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color=color, marker='.', lw=0, label='Model')
    if self.selecAddDSP_PVPlot.isChecked():
      ax2.plot([0, Vmp, Voc], [Isc, Imp, 0], color=color, marker='x', lw=0, label='DataSheet')
    
    # =========== Complementary conditions (LIC)
    S, T, Voc, Isc, Vmp, Imp, Pmp = self.meas_ds3
    [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(S, T, pts)
    
    p3 = ax1.plot(V_model, P_model, label='Low Irradiance Conditions\n{:5.2f} (W/m$^2$)\n{:5.2f} (°C)'.format(S, T-T0))
    color = p3[0].get_color()

    ax1.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color=color, marker='.', lw=0, label='Model')
    if self.selecAddDSP_PVPlot.isChecked():
      ax1.plot([0, Vmp, Voc], [0, Pmp, 0], color=color, marker='x', lw=0, label='DataSheet')

    ax2.plot(V_model, I_model, color=color)
    ax2.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color=color, marker='.', lw=0, label='Model')
    if self.selecAddDSP_PVPlot.isChecked():
      ax2.plot([0, Vmp, Voc], [Isc, Imp, 0], color=color, marker='x', lw=0, label='DataSheet')

    # =========== Temperature behaviour 
    df_temperature = self.df_temperature.iloc[:, self.df_temperature.columns.get_level_values(0).str.contains(current_Model)]
    df_temperature = df_temperature.T.droplevel(0).T

    TS_temperature = df_temperature.iloc[:,7:]
    T_arr      = TS_temperature.iloc[:, TS_temperature.columns.get_level_values(0).str.contains('T')].values.flatten()
    alpha_spec = TS_temperature.iloc[:, TS_temperature.columns.get_level_values(0).str.contains('alpha')].values.flatten()
    beta_spec  = TS_temperature.iloc[:, TS_temperature.columns.get_level_values(0).str.contains('beta')].values.flatten()
    gamma_spec = TS_temperature.iloc[:, TS_temperature.columns.get_level_values(0).str.contains('gamma')].values.flatten()


    df_temp = df_temperature.iloc[:,:6]
    alpha_mean = df_temp.iloc[:, df_temp.columns.get_level_values(0).str.contains('alpha')].values.flatten()
    beta_mean  = df_temp.iloc[:, df_temp.columns.get_level_values(0).str.contains('beta')].values.flatten()
    gamma_mean = df_temp.iloc[:, df_temp.columns.get_level_values(0).str.contains('gamma')].values.flatten()    


    p4 = ax3.plot(T_arr, alpha_spec)
    ax3.plot([alpha_mean[0]], [alpha_mean[1]], 
             color = p4[0].get_color(), marker='D', lw=0)
    if self.selecAddDSP_PVPlot.isChecked():
      ax3.plot([self.meas_ds1[1]-T0], [self.alpha_spec], 
               color = p4[0].get_color(), marker='x', lw=0)

    p5 = ax4.plot(T_arr, beta_spec)
    ax4.plot([beta_mean[0]], [beta_mean[1]], 
             color = p5[0].get_color(), marker='D', lw=0)
    if self.selecAddDSP_PVPlot.isChecked():
      ax4.plot([self.meas_ds1[1]-T0], [self.beta_spec], 
               color = p5[0].get_color(), marker='x', lw=0)
    
    p6 = ax5.plot(T_arr, gamma_spec)
    ax5.plot([gamma_mean[0]], [gamma_mean[1]], 
             color = p6[0].get_color(), marker='D', lw=0)
    if self.selecAddDSP_PVPlot.isChecked():
      ax5.plot([self.meas_ds1[1]-T0], [self.gamma_spec], 
                color = p6[0].get_color(), marker='x', lw=0)
    

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

     # =========== add legend
    handles, labels = ax1.get_legend_handles_labels() 
   
    ax_legend.legend(handles, labels, loc=10, ncol=3)
   

    canvas = FigureCanvas(fig)  
    return canvas
  
  def plot_histogram(self):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
      {
        'font.size': 10,
        "text.usetex": False,
        "font.family": "serif",
        "font.sans-serif": ['Computer Modern'],
      }
    )
    def tick_format(x, y):
      return "%03.2f"%x
    
    idx_error = self.selecDataHist.currentIndex()
    current_model = self.selecModelHist.currentText()         
    tech_name = self.selecTechHist.currentText()         
    df0 = self.df_summary.iloc[:,self.df_summary.columns.get_level_values(0).str.contains(current_model)]
    df0 = df0.iloc[:,~(df0.columns.get_level_values(0).str.contains('Temperature'))]
    df1 = self.df_error.iloc[:, self.df_error.columns.get_level_values(0).str.contains(current_model)]
    if current_model =='11PFF':
      if idx_error <=4:
        dfx = df0.iloc[df0.index.get_level_values(0)==tech_name,(df0.columns.get_level_values(1).str.contains('eF'))]
        dfx = dfx.iloc[:, idx_error]
      else:
        dfx = df1.loc[df1.index.get_level_values(0)==tech_name,:]
        if (idx_error-5)==0:
          dat = 'Vmp'
        elif (idx_error-5)==1:
          dat = 'Imp'
        elif (idx_error-5)==2:
          dat = 'Pmp'
        dfx = dfx.iloc[:,(dfx.columns.get_level_values(2).str.contains(dat))]
        dfx = pd.DataFrame(dfx.to_numpy().flatten(), columns=[dat]).iloc[:, 0]
    elif current_model=='7PFF':
      if idx_error <=2:
        dfx = df0.iloc[df0.index.get_level_values(0)==tech_name,(df0.columns.get_level_values(1).str.contains('Ffp|Ftc|Ftc2'))]
        dfx = dfx.iloc[:, idx_error]
      else:
        dfx = df1.loc[df1.index.get_level_values(0)==tech_name,:]
        if (idx_error-3)==0:
          dat = 'Vmp'
        elif (idx_error-3)==1:
          dat = 'Imp'
        elif (idx_error-3)==2:
          dat = 'Pmp'
        dfx = dfx.iloc[:,(dfx.columns.get_level_values(2).str.contains(dat))]
        dfx = pd.DataFrame(dfx.to_numpy().flatten(), columns=[dat]).iloc[:, 0]
    # remove_outliders
    eF = dfx[dfx.between(dfx.quantile(.05), dfx.quantile(.95))]

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 1)  
    ax = fig.add_subplot(gs[0, 0])  
    sns.histplot(eF, bins=20, kde=False, ax=ax, color='blue', edgecolor='black')
    ax.xaxis.set_major_locator(ticker.LinearLocator(6))
    ax.yaxis.set_major_locator(ticker.LinearLocator(6))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(tick_format))
    ax.set_xlabel(tech_name)
    ax.set_xlim([np.min([ ax.get_xlim()[0], -ax.get_xlim()[1]]), 
                 np.max([-ax.get_xlim()[0],  ax.get_xlim()[1]])])
    canvas = FigureCanvas(fig)  
    return canvas


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














