import os, sys, copy
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QToolBar, QHeaderView, 
                              QTableWidgetItem, QAbstractItemView, QVBoxLayout, 
                              QDialog, QLineEdit, QMessageBox, QWidget, 
                              QStyledItemDelegate, QHBoxLayout, QCheckBox)


from PyQt6.QtGui import QAction, QDoubleValidator, QIntValidator
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6 import uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.constants import zero_Celsius as T0

import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.backend_bases import MouseButton

source_path = os.getcwd()
src_path = os.path.join(source_path, 'src')
assert all([os.path.exists(src_path), os.path.isdir(src_path)])
if src_path not in sys.path:
  sys.path.append(src_path)

from src.utils import *

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
def intersection(x1, y1, x2, y2):
  # Interpolate the curves
  f1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=np.nan)
  f2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value=np.nan)

  # Define the difference function
  def difference(x):
      return f1(x) - f2(x)

  # Find the intersection points
  x_combined = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), 500)  # Overlapping domain
  roots = []
  for i in range(len(x_combined) - 1):
      x_left, x_right = x_combined[i], x_combined[i+1]
      if np.sign(difference(x_left)) != np.sign(difference(x_right)):
          root = root_scalar(difference, bracket=[x_left, x_right], method='brentq')
          if root.converged:
              roots.append(root.root)

  # Get y-values for the intersection points
  xx = np.array(roots)
  yy = f1(xx)
  return xx[0], yy[0]



ui_path = os.path.join(src_path, 'ui')
summary_path = os.path.join(ui_path, 'summary.csv')
main_path = os.path.join(ui_path, 'main.ui')
add_path = os.path.join(ui_path, 'additional.ui')


from src.PVfitting_CEC import PVfitting_11PFF_CEC
getModel = PVfitting_11PFF_CEC()



from PVModel import Model11PFF



class NumericalInputDelegate(QStyledItemDelegate):
  def __init__(self, min_value=None, max_value=None, parent=None):
    super().__init__(parent)
    self.min_value = min_value
    self.max_value = max_value

  def createEditor(self, parent, option, index):
    # Create a QLineEdit editor with a validator for numerical input
    editor = QLineEdit(parent)
    if self.min_value is not None and self.max_value is not None:
        validator = QIntValidator(self.min_value, self.max_value, parent)
    else:
        validator = QIntValidator(parent)  # Allow any integer if no range is specified
    editor.setValidator(validator)
    return editor

class FloatInputDelegate(QStyledItemDelegate):
  def __init__(self, min_value=None, max_value=None, parent=None):
    super().__init__(parent)
    self.min_value = min_value
    self.max_value = max_value

  def createEditor(self, parent, option, index):
    # Create a QLineEdit editor with a QDoubleValidator for float input
    editor = QLineEdit(parent)
    if self.min_value is not None and self.max_value is not None:
        validator = QDoubleValidator(self.min_value, self.max_value, 2, parent)  # 2 decimal places
    else:
        validator = QDoubleValidator(parent)  # Allow any float if no range is specified
    editor.setValidator(validator)
    return editor

class RunWindow(QDialog):
  row_created = pyqtSignal(dict)  
  config = {
      'n_max':3.0, 
      'n_min':0.1, 
      'Rs_min':1e-6,
      'pts_presolve':10, 
      'tol':1e-9,
      'gtol':1e-1,
      'mS_max':3.0,
      'n_Mmax':0.1
  }
  def __init__(self, main_window):
    super().__init__()

    
    # Load the UI file
    uic.loadUi(add_path, self)
    
    self.setWindowTitle("New Module")
    self.setFixedSize(728, 358)
    self.SelectMethod.setDisabled(True)

    width = 90
    self.tableMeasurements.setColumnWidth(0, width)
    self.tableMeasurements.setColumnWidth(1, width)
    self.tableMeasurements.setColumnWidth(2, width)
    self.tableMeasurements.setColumnWidth(3, width)
    self.tableMeasurements.setColumnWidth(4, width)
    self.tableMeasurements.setColumnWidth(5, width)
    self.tableMeasurements.setColumnWidth(6, width)

    header = self.tableMeasurements.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)  
    self.tableMeasurements.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.tableMeasurements.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    # Disable interaction with vertical and horizontal scroll bars
    self.tableMeasurements.verticalScrollBar().setEnabled(False)
    self.tableMeasurements.horizontalScrollBar().setEnabled(False)

    # Connecting the button's click signal to the method
    self.ButtonRun.clicked.connect(self.readData)
    self.row_created.connect(main_window.update_database)

    self.lineEditModule.setAlignment(Qt.AlignmentFlag.AlignRight)
    self.lineEditTechnology.setAlignment(Qt.AlignmentFlag.AlignRight)

    self.lineEditNs.textChanged.connect(lambda x: self.validate_int(self.lineEditNs, x))
    self.lineEditNp.textChanged.connect(lambda x: self.validate_int(self.lineEditNp, x))

    self.lineEditNs.setAlignment(Qt.AlignmentFlag.AlignRight)
    self.lineEditNp.setAlignment(Qt.AlignmentFlag.AlignRight)
    self.lineEditAlpha.setAlignment(Qt.AlignmentFlag.AlignRight)
    self.lineEditBeta.setAlignment(Qt.AlignmentFlag.AlignRight)
    self.lineEditGamma.setAlignment(Qt.AlignmentFlag.AlignRight)
    self.lineEditTol.setAlignment(Qt.AlignmentFlag.AlignRight)

    self.lineEditNs.setValidator(QIntValidator(1, 1000)) 
    self.lineEditNp.setValidator(QIntValidator(1, 1000)) 
    self.lineEditAlpha.setValidator(QDoubleValidator(-5, 5, 10)) 
    self.lineEditBeta.setValidator(QDoubleValidator(-5, 5, 10)) 
    self.lineEditGamma.setValidator(QDoubleValidator(-5, 5, 10)) 
    self.lineEditTol.setValidator(QDoubleValidator(1e-20, 5, 10)) 
    


    self.table_editor = {}
    validator = QDoubleValidator(0.00, 999.99, 5) 
    for row in range(self.tableMeasurements.rowCount()):
      for col in range(self.tableMeasurements.columnCount()):
        line_edit = QLineEdit()
        line_edit.setValidator(validator)
        line_edit.setStyleSheet("background-color: white") 
        line_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.tableMeasurements.setCellWidget(row, col, line_edit)

        col0 = self.tableMeasurements.horizontalHeaderItem(col).text().replace('(V)', '').replace('(A)', '').replace('(W)', '').replace('(W/m2)', '').replace('(°C)', '').replace(' ', '')
        row0 = self.tableMeasurements.verticalHeaderItem(row).text().replace(' ', '')
        self.table_editor[(col0, row0)] = line_edit

    
  def validate_int(self, Widget, text):
    if text:
      if int(text)<=0:
        Widget.setText(text[:-1])

  def close_window(self):
    self.close()

  def readData(self):
    PVModule   = self.lineEditModule.text()
    Technology = self.lineEditTechnology.text()
    Ns = self.lineEditNs.text()
    Np = self.lineEditNp.text()

    alpha = self.lineEditAlpha.text()
    beta  = self.lineEditBeta.text()
    gamma = self.lineEditGamma.text()
    
    S_STC   = self.table_editor[('S', 'STC')].text()
    T_STC   = self.table_editor[('T', 'STC')].text()
    Voc_STC = self.table_editor[('Voc', 'STC')].text()
    Isc_STC = self.table_editor[('Isc', 'STC')].text()
    Vmp_STC = self.table_editor[('Vmp', 'STC')].text()
    Imp_STC = self.table_editor[('Imp', 'STC')].text()
    Pmp_STC = self.table_editor[('Pmp', 'STC')].text()

    S_NOCT   = self.table_editor[('S', 'NOCT')].text()
    T_NOCT   = self.table_editor[('T', 'NOCT')].text()
    Voc_NOCT = self.table_editor[('Voc', 'NOCT')].text()
    Isc_NOCT = self.table_editor[('Isc', 'NOCT')].text()
    Vmp_NOCT = self.table_editor[('Vmp', 'NOCT')].text()
    Imp_NOCT = self.table_editor[('Imp', 'NOCT')].text()
    Pmp_NOCT = self.table_editor[('Pmp', 'NOCT')].text()

    S_LIC   = self.table_editor[('S', 'LIC')].text()
    T_LIC   = self.table_editor[('T', 'LIC')].text()
    Voc_LIC = self.table_editor[('Voc', 'LIC')].text()
    Isc_LIC = self.table_editor[('Isc', 'LIC')].text()
    Vmp_LIC = self.table_editor[('Vmp', 'LIC')].text()
    Imp_LIC = self.table_editor[('Imp', 'LIC')].text()
    Pmp_LIC = self.table_editor[('Pmp', 'LIC')].text()


    cond_general = [len(k)>0 for k in [PVModule, Technology, Ns, Np]]
    cond_thermal = [len(k)>0 for k in [alpha, beta, gamma]]
    cond_STC     = [len(k)>0 for k in [S_STC, T_STC, Voc_STC, Isc_STC, Vmp_STC, Imp_STC, Pmp_STC]]
    cond_NOCT    = [len(k)>0 for k in [S_NOCT, T_NOCT, Vmp_NOCT, Imp_NOCT, Pmp_NOCT]]
    cond_LIC     = [len(k)>0 for k in [S_LIC, T_LIC, Vmp_LIC, Imp_LIC, Pmp_LIC]]

    if all(cond_general):
      Ns = int(Ns)
      Np = int(Np)
      if all(cond_thermal):
        alpha = float(alpha)
        beta  = float(beta)
        gamma = float(gamma)
        if all(cond_STC):
          S_STC   = float(S_STC)
          T_STC   = float(T_STC)+T0
          Vmp_STC = float(Vmp_STC)
          Imp_STC = float(Imp_STC)
          Pmp_STC = float(Pmp_STC)
          Voc_STC = float(Voc_STC)
          Isc_STC = float(Isc_STC)
          VI_STC = [Voc_STC>Vmp_STC, Isc_STC>Imp_STC]
          if all(VI_STC):
            if all(cond_NOCT):
              S_NOCT   = float(S_NOCT)
              T_NOCT   = float(T_NOCT)+T0
              Vmp_NOCT = float(Vmp_NOCT)
              Imp_NOCT = float(Imp_NOCT)
              Pmp_NOCT = float(Pmp_NOCT)

              VI_NOCT = [True]
              if len(Voc_NOCT)>0:
                Voc_NOCT = float(Voc_NOCT)
                VI_NOCT.append([Voc_NOCT>Vmp_NOCT])
              else:
                Voc_NOCT = np.nan
              if len(Isc_NOCT)>0:
                Isc_NOCT = float(Isc_NOCT)
                VI_NOCT.append([Isc_NOCT>Imp_NOCT])
              else:
                Isc_NOCT = np.nan
              if all(VI_NOCT):



                if all(cond_LIC):
                  S_LIC   = float(S_LIC)
                  T_LIC   = float(T_LIC)+T0
                  Vmp_LIC = float(Vmp_LIC)
                  Imp_LIC = float(Imp_LIC)
                  Pmp_LIC = float(Pmp_LIC)
                  VI_LIC = [True]
                  if len(Voc_LIC)>0:
                    Voc_LIC = float(Voc_LIC)
                    VI_LIC.append([Voc_LIC>Vmp_LIC])
                  else:
                    Voc_LIC = np.nan
                  if len(Isc_LIC)>0:
                    Isc_LIC = float(Isc_LIC)
                    VI_LIC.append([Isc_LIC>Imp_LIC])
                  else:
                    Isc_LIC = np.nan
                  if all(VI_LIC):
                    if len(self.lineEditTol.text())>0:
                      self.config['tol'] = float(self.lineEditTol.text())




                      try:
                        
                        print('aaa')
                        config = copy.deepcopy(**self.config)
                        print(config)
                        while config['n_max']> config['n_min']:
                          print(config['n_max'])
                          try:
                            self.df_sum = getModel.run(PVModule, 
                                                    Technology,
                                                    Ns, Np, 
                                                    alpha, beta, gamma,
                                                    S_STC, T_STC, Voc_STC, Isc_STC, Vmp_STC, Imp_STC, Pmp_STC,
                                                    S_NOCT, T_NOCT, Voc_NOCT, Isc_NOCT, Vmp_NOCT, Imp_NOCT, Pmp_NOCT,
                                                    S_LIC, T_LIC, Voc_LIC, Isc_LIC, Vmp_LIC, Imp_LIC, Pmp_LIC,
                                                    **config)
                            self.row_created.emit(self.df_sum.to_dict())
                            self.close()
                          except:
                            config['n_max'] -= 0.1


                      except:
                        QMessageBox.warning(self, 'Solver', 'Solver: Model does not converge')
                    else:
                      QMessageBox.warning(self, 'Solver Options', 'Solver Options: Setting the tolerance')
                  
                  
                  else:
                    if not VI_LIC[1]:
                      QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: Voc<Vmp')
                    elif not VI_LIC[2]:
                      QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: Isc<Imp')
                else:
                  if not cond_LIC[0]:
                    QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: No entry “S”')
                  elif not cond_LIC[1]:
                    QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: No entry “T”')
                  elif not cond_LIC[2]:
                    QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: No entry “Vmp”')
                  elif not cond_LIC[3]:
                    QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: No entry “Imp”')
                  elif not cond_LIC[4]:
                    QMessageBox.warning(self, 'Measurements LIC', 'Measurements LIC: No entry “Pmp”')


              else:
                if not VI_NOCT[1]:
                  QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: Voc<Vmp')
                elif not VI_NOCT[2]:
                  QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: Isc<Imp')
            else:
              if not cond_NOCT[0]:
                QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: No entry “S”')
              elif not cond_NOCT[1]:
                QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: No entry “T”')
              elif not cond_NOCT[2]:
                QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: No entry “Vmp”')
              elif not cond_NOCT[3]:
                QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: No entry “Imp”')
              elif not cond_NOCT[4]:
                QMessageBox.warning(self, 'Measurements NOCT', 'Measurements NOCT: No entry “Pmp”')



          else:
            if not VI_STC[0]:
              QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: Voc<Vmp')
            elif not VI_STC[1]:
              QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: Isc<Imp')
        else:
          if not cond_STC[0]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “S”')
          elif not cond_STC[1]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “T”')
          elif not cond_STC[2]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “Voc”')
          elif not cond_STC[3]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “Isc”')
          elif not cond_STC[4]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “Vmp”')
          elif not cond_STC[5]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “Imp”')
          elif not cond_STC[6]:
            QMessageBox.warning(self, 'Measurements STC', 'Measurements STC: No entry “Pmp”')


      else:
        if not cond_thermal[0]:
          QMessageBox.warning(self, 'Thermal parameters', 'Thermal parameters: No entry “alpha”')
        elif not cond_thermal[1]:
          QMessageBox.warning(self, 'Thermal parameters', 'Thermal parameters: No entry “beta”')
        elif not cond_thermal[2]:
          QMessageBox.warning(self, 'Thermal parameters', 'Thermal parameters: No entry “gamma”')


    else:
      if not cond_general[0]:
        QMessageBox.warning(self, 'General parameters', 'General parameters: No entry “PVModule”')
      elif not cond_general[1]:
        QMessageBox.warning(self, 'General parameters', 'General parameters: No entry “Technology”')
      elif not cond_general[2]:
        QMessageBox.warning(self, 'General parameters', 'General parameters: No entry “Ns”')
      elif not cond_general[3]:
        QMessageBox.warning(self, 'General parameters', 'General parameters: No entry “Np”')

class MainWindow(QMainWindow):
  close_all = pyqtSignal()  # Define a signal

  STC_marker_options  = None
  NOCT_marker_options = None
  LIC_marker_options  = None

  STC_marker  = None
  NOCT_marker = None
  LIC_marker  = None

  STC_annotation = None
  NOCT_annotation = None
  LIC_annotation = None
  
  STC_marker0  = None
  NOCT_marker0 = None
  LIC_marker0  = None
  user_marker0 = None
  
  STC_curve  = None
  NOCT_curve = None
  LIC_curve  = None
  user_curve = None


  LoadData = {}

  
  
  



  def __init__(self):
    super().__init__()

    # database inicialization
    self.df_summary = pd.read_csv(summary_path, index_col=[0,1], header=[0,1,2])
    self.df_summary.columns = pd.MultiIndex.from_tuples([replace_unnamed(level) for level in self.df_summary.columns])
    self.filter_11PFF    = self.df_summary.columns.get_level_values(0).str.contains('11PFF')
    self.filter_converge = self.df_summary.columns.get_level_values(1).str.contains('converge')
    def task(df):
      return df['Module'].values.tolist()
    self.df_summary.index.names = ['Technology', 'Module']
    df = self.df_summary.iloc[:, self.filter_11PFF&self.filter_converge]
    df = df.T.droplevel(0).droplevel(1).T
    df = df.loc[df.iloc[:,0].astype(bool)].drop(columns=df.columns)
    df = df.reset_index().groupby('Technology').apply(task, include_groups=False)
    self.Tech2Module = df.to_dict()

    # Load the UI file
    uic.loadUi(main_path, self)
    self.setWindowTitle("Parameter identification")

    # Find the toolbar and set the custom styles
    self.toolbar = self.findChild(QToolBar, "toolbar")
    self.toolbar.setMovable(False)
    self.toolbar.setFloatable(False)

    # Set fixed window size
    self.setFixedSize(1620, 720+200) 
    self.Options.setFixedWidth(330)
    self.Tables.setFixedWidth(330)
    
    # Initialize additional window
    self.run_window = RunWindow(self)




    # General config
    # ===========================================
    self.SelectTechnology.addItems(self.Tech2Module.keys())
    self.SelectModule.addItems(self.Tech2Module[self.SelectTechnology.currentText()])
    self.CheckBoxDataSheet.setChecked(True)
    self.CheckBoxUser.setChecked(False)  

    # Set the table size
    self.tableParams.setRowCount(len(self.k11PFF))
    self.tableParams.setColumnWidth(0, 216)
    self.set_tables_properties(self.tableParams)

    # Set the table size
    self.tableDataSheet.setColumnWidth(0, 78)
    self.tableDataSheet.setColumnWidth(1, 79)
    self.tableDataSheet.setColumnWidth(2, 78)
    self.set_tables_properties(self.tableDataSheet)

    # Set the table size
    self.tableDataSheet2.setColumnWidth(0, 206)
    self.set_tables_properties(self.tableDataSheet2)




















    # Set the table size
    self.tableLoads.setColumnWidth(0, 129)
    self.tableLoads.setColumnWidth(1, 69)
    self.tableLoads.setColumnWidth(2, 60)

    header = self.tableLoads.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)  
    self.tableLoads.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.tableLoads.horizontalScrollBar().setEnabled(False)
    
    layout = self.findChild(QVBoxLayout, "verticalLayout2")
    layout.setContentsMargins(0, 12, 0, 0) 




    # Add checkboxes to the second column and center them
    for row in range(self.tableLoads.rowCount()):
      checkbox = QCheckBox()
      checkbox_widget = QWidget()
      checkbox_widget.setStyleSheet("background-color: rgb(255, 255, 255);")  
      layout = QHBoxLayout(checkbox_widget)
      layout.addWidget(checkbox)
      layout.setAlignment(checkbox, Qt.AlignmentFlag.AlignCenter) 
      layout.setContentsMargins(0, 0, 0, 0)  
      self.tableLoads.setCellWidget(row, 2, checkbox_widget)
      checkbox.stateChanged.connect(self.updateGraph)

      

      self.LoadData[row] = {
        "checkbox":checkbox,
        "marker0": None,
        "annotation0": None,
        "annotation": None,
        "curve": None,
      }
    
    # Apply the numerical input delegate to the third column
    self.tableLoads.setItemDelegateForColumn(1, FloatInputDelegate(0, 1e6, self))






    











    # linking functions with graphic actions
    self.SelectTechnology.currentIndexChanged.connect(self.ComboxFillModules)
    self.SelectModule.currentIndexChanged.connect(self.compile)
    self.ButtonUpdate.clicked.connect(self.updateGraph)
    self.CheckBoxDataSheet.clicked.connect(self.update_marker)
    self.CheckBoxUser.clicked.connect(self.updateGraph)
    self.CheckBoxTag.clicked.connect(self.update_tag)
    self.tabCurve.currentChanged.connect(self.updateGraph)

    # Connect the toolbar actions
    openRunWindows = self.findChild(QAction, "customAction")
    openRunWindows.triggered.connect(self.openRunWindows)

    # ==========================================
    # initial render
    self._init_operator_chart()
    self._init_plot_chart()
    self.compile()
    self.updateGraph()

  def _init_operator_chart(self):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
      {
        'font.size': 10,
        "text.usetex": False,
        "font.family": "serif",
        
        "grid.color": "0.95",
        "grid.linestyle": "--",
      }
    )
    figure = Figure(tight_layout=False)
    self.canvasOptions = FigureCanvas(figure)
    
    layout = QVBoxLayout()
    layout.addWidget(self.canvasOptions)
    self.setLayout(layout)
    layout.setContentsMargins(0, 13, 0, 0)
    self.plotContainerOptions.setLayout(layout)

    self.axOptions = figure.add_subplot(111)
    self.canvasOptions.mpl_connect('button_press_event', self.onclickMap)
    
    # Create a Path object
    self.polygon = Path(self.polygon_points)  
    self.axOptions.add_patch(patches.Polygon(self.polygon_points, closed=True, fill=True, facecolor='black', edgecolor='r', alpha=0.1))
    
    # Plot the polygon and add markers to the vertices
    self.axOptions.plot(*zip(*self.polygon_points), marker='.', color='black', linestyle='-')  

    self.last_marker = None
    self.last_vline = None
    self.last_hline = None
    self.last_annotation = None 
    self.user_annotation = None 
    
    self.user_T = 55
    self.user_S = 600
    self.update_marker_select((self.user_T, self.user_S), '*')

    # limits
    self.axOptions.set_xlim([0, 85])
    self.axOptions.set_ylim([0, 1200])
  
    # Set labels
    self.axOptions.set_xlabel('Temperature (°C)')
    self.axOptions.set_title(r'A', color="white")
    self.axOptions.set_ylabel(r'Irradiance (W/m$^2$)', rotation=0, ha='right', va='top')
    self.axOptions.get_yaxis().set_label_coords(0.37, 1.1)


    # ==========================================
    plt.rcParams.update({"axes.grid" : True})
    figure = Figure(tight_layout=False)
    self.canvasPV = FigureCanvas(figure)
    layout = QVBoxLayout()
    layout.addWidget(self.canvasPV)
    layout.setContentsMargins(0, 0, 0, 0)
    self.setLayout(layout)
    self.plotContainerPV.setLayout(layout)
    
    gs = gridspec.GridSpec(2, 1, height_ratios=(1, 9))
    self.axPV = figure.add_subplot(gs[1:, :])
    self.axPV.set_xlabel('Voltage (V)')
    self.axPV.set_ylabel('Power (W)')
    figure.tight_layout()

    self.axPV.xaxis.set_major_locator(ticker.LinearLocator(6))
    self.axPV.yaxis.set_major_locator(ticker.LinearLocator(6))
    self.axPV.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,y: "%03.1f"%x))
    self.axPV.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,y: "%03.1f"%x))
    self.axPV.set_xlim(left=0)
    self.axPV.set_ylim(bottom=0)

    # legend
    self.axPV_legend = figure.add_subplot(gs[0, :])
    self.axPV_legend.axis('off')

  def _init_plot_chart(self):
    plt.rcParams.update({"axes.grid" : True})
    figure = Figure(tight_layout=False)
    self.canvasIV = FigureCanvas(figure)
    layout = QVBoxLayout()
    layout.addWidget(self.canvasIV)
    layout.setContentsMargins(0, 0, 0, 0)
    self.setLayout(layout)
    self.plotContainerIV.setLayout(layout)
          
    gs = gridspec.GridSpec(2, 1, height_ratios=(1, 9))
    self.axIV = figure.add_subplot(gs[1:, :])
    self.axIV.set_xlabel('Voltage (V)')
    self.axIV.set_ylabel('Current (A)')
    figure.tight_layout()

    self.axIV.xaxis.set_major_locator(ticker.LinearLocator(6))
    self.axIV.yaxis.set_major_locator(ticker.LinearLocator(6))
    self.axIV.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,y: "%03.1f"%x))
    self.axIV.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,y: "%03.1f"%x))
    self.axIV.set_xlim(left=0)
    self.axIV.set_ylim(bottom=0)

    # legend
    self.axIV_legend = figure.add_subplot(gs[0, :])
    self.axIV_legend.axis('off')

  def update_database(self, new_row=None):
    def task(df):
      return df['Module'].values.tolist()
    
    if isinstance(new_row, dict):
      df_new_row = pd.DataFrame(new_row)
      self.df_summary = pd.concat([self.df_summary, df_new_row])
      Technology, Module = pd.DataFrame(df_new_row).index.to_list()[0]
      self.df_summary.index.names = ['Technology', 'Module']
      self.df_summary.to_csv(summary_path)

      df = self.df_summary.iloc[:, self.filter_11PFF&self.filter_converge]
      df = df.T.droplevel(0).droplevel(1).T
      df = df.loc[df.iloc[:,0].astype(bool)].drop(columns=df.columns)
      df = df.reset_index().groupby('Technology').apply(task, include_groups=False)
      self.Tech2Module = df.to_dict()

      self.SelectTechnology.currentIndexChanged.disconnect(self.ComboxFillModules)
      self.SelectTechnology.clear()
      self.SelectTechnology.currentIndexChanged.connect(self.ComboxFillModules)
      self.SelectTechnology.addItems(self.Tech2Module.keys())
      self.SelectTechnology.setCurrentText(Technology) 
    
      # reset children combox
      self.SelectModule.currentIndexChanged.disconnect(self.compile)
      self.SelectModule.clear()
      self.SelectModule.currentIndexChanged.connect(self.compile)
      self.SelectModule.addItems(self.Tech2Module[Technology])
      self.SelectModule.setCurrentText(Module) 

      # compile
      self.compile()
      self.updateGraph()


      
      















  def openRunWindows(self):
    self.run_window.exec()
    self.close_all.connect(self.run_window.close_window)

  def updateGraph(self):
    mPFF  = Model11PFF(*self.dfp[self.k11PFF].values.tolist()[0])
    index = self.tabCurve.currentIndex()
    Curve = self.tabCurve.tabText(index)
    
    self.update_curve(Curve, mPFF)
    self.update_marker()
    self.update_table()
    self.update_tag()

  def update_curve(self, Curve, mPFF, pts:int=100):
    if self.STC_curve:
      try:
        self.STC_curve.remove()
      except:
        pass
    if self.NOCT_curve:
      try:
        self.NOCT_curve.remove()
      except:
        pass
    if self.LIC_curve:
      try:
        self.LIC_curve.remove()
      except:
        pass
    if self.user_curve:
      try:
        self.user_curve.remove()
      except:
        pass

    if self.STC_marker0:
      try:
        self.STC_marker0.remove()
      except:
        pass
    if self.NOCT_marker0:
      try:
        self.NOCT_marker0.remove()
      except:
        pass
    if self.LIC_marker0:
      try:
        self.LIC_marker0.remove()
      except:
        pass
    if self.user_marker0:
      try:
        self.user_marker0.remove()
      except:
        pass
    for row in range(self.tableLoads.rowCount()):
      try:
        self.LoadData[row]["curve"].remove()
      except:
        pass
      try:
          self.LoadData[row]["marker0"].remove()
      except:
        pass
      try:
          self.LoadData[row]["annotation0"].remove()
      except:
        pass




    
    labels = []
    xmax = 0
    self.ymax = 0
    # =========== Reference conditions
    S, T, Voc, Isc, Vmp, Imp, Pmp = self.STCm
    [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(S, T, pts)
    labels.append('Standar Test Condition\n{:5.2f}(W/m$^2$)\n{:5.2f}(°C)'.format(S, T-T0))
    if Curve == "Curve P-V":
      self.STC_curve,   = self.axPV.plot(V_model, P_model, color='tab:blue')
      self.STC_marker0, = self.axPV.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color='tab:blue',
                                       marker='.', lw=0, label='Model')
      if self.ymax<Pmp_m:
        self.ymax = Pmp_m
    elif Curve == "Curve I-V":
      self.STC_curve,   = self.axIV.plot(V_model, I_model, color='tab:blue')
      self.STC_marker0, = self.axIV.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color='tab:blue', 
                                       marker='.', lw=0, label='Model')
      if self.ymax<Isc_m:
        self.ymax = Isc_m
    if xmax<Voc_m:
      xmax = Voc_m


    # =========== Complementary conditions (NOCT)
    S, T, Voc, Isc, Vmp, Imp, Pmp = self.NOCTm
    [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(S, T, pts)
    labels.append('Nominal Operating Cell Temperature\n{:5.2f} (W/m$^2$)\n{:5.2f} (°C)'.format(S, T-T0))
    if Curve == "Curve P-V":
      self.NOCT_curve,   = self.axPV.plot(V_model, P_model, color='tab:orange')
      self.NOCT_marker0, = self.axPV.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color='tab:orange', marker='.', lw=0)
      if self.ymax<Pmp_m:
        self.ymax = Pmp_m
    elif Curve == "Curve I-V":
      self.NOCT_curve,   = self.axIV.plot(V_model, I_model,  color='tab:orange')
      self.NOCT_marker0, = self.axIV.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color='tab:orange', marker='.', lw=0)
      if self.ymax<Isc_m:
        self.ymax = Isc_m
    if xmax<Voc_m:
      xmax = Voc_m

    # =========== Complementary conditions (LIC)
    S, T, Voc, Isc, Vmp, Imp, Pmp = self.LICm
    [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(S, T, pts)
    labels.append('Low Irradiance Conditions\n{:5.2f} (W/m$^2$)\n{:5.2f} (°C)'.format(S, T-T0))
    if Curve == "Curve P-V":
      self.LIC_curve,   = self.axPV.plot(V_model, P_model,  color='tab:green')
      self.LIC_marker0, = self.axPV.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color='tab:green', marker='.', lw=0)
      if self.ymax<Pmp_m:
        self.ymax = Pmp_m
    elif Curve == "Curve I-V":
      self.LIC_curve,   = self.axIV.plot(V_model, I_model,  color='tab:green')
      self.LIC_marker0, = self.axIV.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color='tab:green', marker='.', lw=0)
      if self.ymax<Isc_m:
        self.ymax = Isc_m
    if xmax<Voc_m:
      xmax = Voc_m


    # =========== User Plot
    if self.CheckBoxUser.isChecked():
      try:
        [[Isc_m, Vsc_m, Imp_m, Vmp_m, Pmp_m, Ioc_m, Voc_m], [V_model, I_model, P_model]] = mPFF(self.user_S, self.user_T+T0, pts)
        if Curve == "Curve P-V":
          self.user_curve,   = self.axPV.plot(V_model, P_model, color='black')
          self.user_marker0, = self.axPV.plot([0, Vmp_m, Voc_m], [0, Pmp_m, 0], color='black', marker='.', lw=0)
          if self.ymax<Pmp_m:
            self.ymax = Pmp_m
          self.user_point = [Vmp_m, Pmp_m]
        elif Curve == "Curve I-V":
          self.user_curve,   = self.axIV.plot(V_model, I_model, color='black')
          self.user_marker0, = self.axIV.plot([0, Vmp_m, Voc_m], [Isc_m, Imp_m, 0], color='black', marker='.', lw=0)
          if self.ymax<Isc_m:
            self.ymax = Isc_m
          self.user_point = [Vmp_m, Imp_m]
        if xmax<Voc_m:
          xmax = Voc_m
      except Exception as e:
        print(e)
        pass


    # =========== add legend
    if Curve == "Curve P-V":
      self.axPV_legend.legend([self.STC_curve, self.NOCT_curve, self.LIC_curve], labels, loc=10, ncol=3)
      self.axPV.set_xlim(right=xmax*1.1)
      self.axPV.set_ylim(top=self.ymax*1.1)
    elif Curve == "Curve I-V":
      self.axIV_legend.legend([self.STC_curve, self.NOCT_curve, self.LIC_curve], labels, loc=10, ncol=3)
      self.axIV.set_xlim(right=xmax*1.1)
      self.axIV.set_ylim(top=self.ymax*1.1)
    

    # =========== resistence
    for row in range(self.tableLoads.rowCount()):
      if self.LoadData[row]["checkbox"].isChecked():
        cell_item = self.tableLoads.item(row, 1)
        if cell_item:
          Rvalue = float(cell_item.text())
          ROW_label = self.tableLoads.verticalHeaderItem(row).text()

          V_arr = np.linspace(0, xmax*1.1, pts*2)
          I_arr = V_arr/Rvalue
          P_arr = V_arr*I_arr

          # STC
          S, T, Voc, Isc, Vmp, Imp, Pmp = self.STCm
          [_, [V_model, I_model, P_model]] = mPFF(S, T, pts)
          x_STC, y_STC = intersection(V_model, I_model, V_arr, I_arr)
          
          # NOCT
          S, T, Voc, Isc, Vmp, Imp, Pmp = self.NOCTm
          [_, [V_model, I_model, P_model]] = mPFF(S, T, pts)
          x_NOCT, y_NOCT = intersection(V_model, I_model, V_arr, I_arr)

          # LIC
          S, T, Voc, Isc, Vmp, Imp, Pmp = self.LICm
          [_, [V_model, I_model, P_model]] = mPFF(S, T, pts)
          x_LIC, y_LIC = intersection(V_model, I_model, V_arr, I_arr)

          # User 
          if self.CheckBoxUser.isChecked():
            [_, [V_model, I_model, P_model]] = mPFF(self.user_S, self.user_T+T0, pts)
            x_user, y_user = intersection(V_model, I_model, V_arr, I_arr)
          else:
            x_user, y_user = [np.nan, np.nan]



          try:
            if Curve == "Curve P-V":
              self.LoadData[row]["curve"],   = self.axPV.plot(V_arr, P_arr, color='tab:red')
              self.LoadData[row]["marker0"], = self.axPV.plot([x_STC, x_NOCT, x_LIC, x_user], 
                                                              [x_STC*y_STC, x_NOCT*y_NOCT, x_LIC*y_LIC, x_user*y_user], 
                                                              color='tab:red', marker='D', lw=0)
              V_arr = V_arr[P_arr<=self.ymax]
              P_arr = P_arr[P_arr<=self.ymax]
              point = [max(V_arr), max(P_arr)]
              
            elif Curve == "Curve I-V":
              self.LoadData[row]["curve"],   = self.axIV.plot(V_arr, I_arr, color='tab:red')
              self.LoadData[row]["marker0"], = self.axIV.plot([x_STC, x_NOCT, x_LIC, x_user], 
                                                              [y_STC, y_NOCT, y_LIC, y_user], 
                                                              color='tab:red', marker='D', lw=0)
              V_arr = V_arr[I_arr<=self.ymax]
              I_arr = I_arr[I_arr<=self.ymax]
              point = [max(V_arr), max(I_arr)]

            # Dynamic annotation positioning
            text_x_offset = 0
            if point[0] >= xmax*0.7:
              text_x_offset = -60
              
            if Curve == "Curve P-V":
              self.LoadData[row]["annotation0"] = self.axPV.annotate(f'{ROW_label}={Rvalue:.2f}',
                                                      xy=point, 
                                                      xytext=(text_x_offset, -5),
                                                      textcoords='offset points',
                                                      color='tab:red',)
            elif Curve == "Curve I-V":
              self.LoadData[row]["annotation0"] = self.axIV.annotate(f'{ROW_label}={Rvalue:.2f}',
                                                      xy=point, 
                                                      xytext=(text_x_offset, -5),
                                                      textcoords='offset points',
                                                      color='tab:red',)
          except:
            pass
    if Curve == "Curve P-V":
      self.canvasPV.draw()
    elif Curve == "Curve I-V":
      self.canvasIV.draw()

  def update_marker(self):
    index = self.tabCurve.currentIndex()
    Curve = self.tabCurve.tabText(index)

    if self.STC_marker_options:
      try:
        self.STC_marker_options.remove()
      except:
        pass
    if self.NOCT_marker_options:
      try:
        self.NOCT_marker_options.remove()
      except:
        pass
    if self.LIC_marker_options:
      try:
        self.LIC_marker_options.remove()
      except:
        pass
  
    if self.CheckBoxDataSheet.isChecked():
      self.STC_marker_options,  = self.axOptions.plot(self.STCm[1]-T0,  self.STCm[0],  marker='x', color='tab:blue')
      self.NOCT_marker_options, = self.axOptions.plot(self.NOCTm[1]-T0, self.NOCTm[0], marker='x', color='tab:orange')
      self.LIC_marker_options,  = self.axOptions.plot(self.LICm[1]-T0,  self.LICm[0],  marker='x', color='tab:green')
    self.canvasOptions.draw()

    if self.STC_marker:
      try:
        self.STC_marker.remove()
      except:
        pass
    if self.NOCT_marker:
      try:
        self.NOCT_marker.remove()
      except:
        pass
    if self.LIC_marker:
      try:
        self.LIC_marker.remove()
      except:
        pass

    if self.CheckBoxDataSheet.isChecked():
      S, T, Voc, Isc, Vmp, Imp, Pmp = self.STCm
      if Curve == "Curve P-V":
        self.STC_marker,  = self.axPV.plot([0, Vmp, Voc],  [0, Pmp, 0], marker='x', color='tab:blue', lw=0, label='DataSheet')
      elif Curve == "Curve I-V":
        self.STC_marker,  = self.axIV.plot([0, Vmp, Voc],  [Isc, Imp, 0], marker='x', color='tab:blue', lw=0, label='DataSheet')

      S, T, Voc, Isc, Vmp, Imp, Pmp = self.NOCTm
      if Curve == "Curve P-V":
        self.NOCT_marker, = self.axPV.plot([0, Vmp, Voc], [0, Pmp, 0], marker='x', color='tab:orange', lw=0, label='DataSheet')
      elif Curve == "Curve I-V":
        self.NOCT_marker, = self.axIV.plot([0, Vmp, Voc], [Isc, Imp, 0], marker='x', color='tab:orange', lw=0, label='DataSheet')

      S, T, Voc, Isc, Vmp, Imp, Pmp = self.LICm
      if Curve == "Curve P-V":
        self.LIC_marker,  = self.axPV.plot([0, Vmp, Voc],  [0, Pmp, 0],  marker='x', color='tab:green', lw=0, label='DataSheet')
      elif Curve == "Curve I-V":
        self.LIC_marker,  = self.axIV.plot([0, Vmp, Voc],  [Isc, Imp, 0],  marker='x', color='tab:green', lw=0, label='DataSheet')

    if Curve == "Curve P-V":
      self.canvasPV.draw()
    elif Curve == "Curve I-V":
      self.canvasIV.draw()

  def update_tag(self):
    index = self.tabCurve.currentIndex()
    Curve = self.tabCurve.tabText(index)
    if self.STC_annotation:
      try:
        self.STC_annotation.remove()
      except:
        pass
    if self.NOCT_annotation:
      try:
        self.NOCT_annotation.remove()
      except:
        pass
    if self.LIC_annotation:
      try:
        self.LIC_annotation.remove()
      except:
        pass
    if self.user_annotation:
      try:
        self.user_annotation.remove()
      except:
        pass

    if self.CheckBoxTag.isChecked():
      S, T, Voc, Isc, Vmp, Imp, Pmp = self.STCm
      if Curve == "Curve P-V":
        point = [Vmp, Pmp]
        self.STC_annotation = self.axPV.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                    xy=point, 
                                                    xytext=(15,15),
                                                    textcoords='offset points',
                                                    color='tab:blue',
                                                    arrowprops=dict(arrowstyle='->', 
                                                                    connectionstyle='arc3,rad=.2',
                                                                    facecolor='tab:blue',
                                                                    ec="tab:blue"))
      elif Curve == "Curve I-V":
        point = [Vmp, Imp]
        self.STC_annotation = self.axIV.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                    xy=point, 
                                                    xytext=(15,15),
                                                    textcoords='offset points',
                                                    color='tab:blue',
                                                    arrowprops=dict(arrowstyle='->', 
                                                                    connectionstyle='arc3,rad=.2',
                                                                    facecolor='tab:blue',
                                                                    ec="tab:blue"))

      S, T, Voc, Isc, Vmp, Imp, Pmp = self.NOCTm
      if Curve == "Curve P-V":
        point = [Vmp, Pmp]
        self.NOCT_annotation = self.axPV.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                    xy=point, 
                                                    xytext=(15,15),
                                                    textcoords='offset points',
                                                    color='tab:orange',
                                                    arrowprops=dict(arrowstyle='->', 
                                                                    connectionstyle='arc3,rad=.2',
                                                                    facecolor='tab:orange',
                                                                    ec="tab:orange"))
      elif Curve == "Curve I-V":
        point = [Vmp, Imp]
        self.NOCT_annotation = self.axIV.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                    xy=point, 
                                                    xytext=(15,15),
                                                    textcoords='offset points',
                                                    color='tab:orange',
                                                    arrowprops=dict(arrowstyle='->', 
                                                                    connectionstyle='arc3,rad=.2',
                                                                    facecolor='tab:orange',
                                                                    ec="tab:orange"))
      
      S, T, Voc, Isc, Vmp, Imp, Pmp = self.LICm
      if Curve == "Curve P-V":
        point = [Vmp, Pmp]
        self.LIC_annotation = self.axPV.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                    xy=point, 
                                                    xytext=(15,15),
                                                    textcoords='offset points',
                                                    color='tab:green',
                                                    arrowprops=dict(arrowstyle='->', 
                                                                    connectionstyle='arc3,rad=.2',
                                                                    facecolor='tab:green',
                                                                    ec="tab:green"))
      elif Curve == "Curve I-V":
        point = [Vmp, Imp]
        self.LIC_annotation = self.axIV.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                    xy=point, 
                                                    xytext=(15,15),
                                                    textcoords='offset points',
                                                    color='tab:green',
                                                    arrowprops=dict(arrowstyle='->', 
                                                                    connectionstyle='arc3,rad=.2',
                                                                    facecolor='tab:green',
                                                                    ec="tab:green"))
    # =========== User Plot
    if all([self.CheckBoxUser.isChecked(),self.CheckBoxTag.isChecked()]):
      try:
        # Add an annotation close to the marker
        if Curve == "Curve P-V":
          self.user_annotation = self.axPV.annotate(f'({self.user_point[0]:.2f}, {self.user_point[1]:.2f})',
                                                      xy=self.user_point, 
                                                      xytext=(15,15),
                                                      textcoords='offset points',
                                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        elif Curve == "Curve I-V":
          self.user_annotation = self.axIV.annotate(f'({self.user_point[0]:.2f}, {self.user_point[1]:.2f})',
                                                      xy=self.user_point, 
                                                      xytext=(15,15),
                                                      textcoords='offset points',
                                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
      except Exception as e:
        print(e)
        pass
    
    if Curve == "Curve P-V":
      self.canvasPV.draw()
    elif Curve == "Curve I-V":
      self.canvasIV.draw()

  def update_table(self):
    self.tableDataSheet2.setHorizontalHeaderLabels([self.Module])
    self.tableParams.setHorizontalHeaderLabels([self.Module])
    for idd, col in enumerate(self.k11PFF):
      value = self.dfp[col].values[0]  
      if col in ['T_ref', 'S_ref']:
        value = "{:.2f}".format(float(value))
      elif col =='I0_ref':
        value = "{:.3e}".format(float(value))
      else:
        if col == 'IL_ref':
          if value<1:
            value = "{:.3e}".format(float(value))
          else:
            value = "{:.5f}".format(float(value))
        else:
          value = "{:.5f}".format(float(value))
        
      item = QTableWidgetItem(str(value))
      item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
      self.tableParams.setItem(idd, 0, item)
      



    for idd, value in enumerate(self.STCm):
      if idd==1:
        value = float(value) - T0
      try:
        value = "{:.2f}".format(float(value))
      except:
        value = str(value)
      item = QTableWidgetItem(value)
      item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
      self.tableDataSheet.setItem(idd, 0, item)

    for idd, value in enumerate(self.NOCTm):
      if idd==1:
        value = float(value) - T0
      try:
        value = "{:.2f}".format(float(value))
      except:
        value = str(value)
      item = QTableWidgetItem(value)
      item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
      self.tableDataSheet.setItem(idd, 1, item)

    for idd, value in enumerate(self.LICm):
      if idd==1:
        value = float(value) - T0
      try:
        value = "{:.2f}".format(float(value))
      except:
        value = str(value)
      item = QTableWidgetItem(value)
      item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
      self.tableDataSheet.setItem(idd, 2, item)

    for idd, value in enumerate(self.dfDS.iloc[:, -5:].values[0].tolist()):
      if idd<=2:
        value = "{:.8f}".format(float(value))
      else:
        value = "{:4d}".format(int(value))
      item = QTableWidgetItem(value)
      item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
      self.tableDataSheet2.setItem(idd, 0, item)

  def set_tables_properties(self, table):
    # Assuming self.tableWidget is your QTableWidget
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)  # Disable user resizing
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    # Disable interaction with vertical and horizontal scroll bars
    table.verticalScrollBar().setEnabled(False)
    table.horizontalScrollBar().setEnabled(False)

  def closeEvent(self, event):
      self.close_all.emit()  # Emit the signal when the main window is closed
      super().closeEvent(event)

  def ComboxFillModules(self):
    Technology = self.SelectTechnology.currentText() 

    # reset children combox
    self.SelectModule.currentIndexChanged.disconnect(self.compile)
    self.SelectModule.clear()
    self.SelectModule.currentIndexChanged.connect(self.compile)
    self.SelectModule.addItems(self.Tech2Module[Technology])

  def compile(self):
    self.Technology = self.SelectTechnology.currentText()
    self.Module     = self.SelectModule.currentText()
    
    # get row
    dfs = self.df_summary.loc[[(self.Technology, self.Module)],:]    

    # get datasheet 
    self.dfDS = dfs.iloc[:,dfs.columns.get_level_values(0).str.contains('DataSheet')]
    self.dfDS = self.dfDS.T.droplevel(0).T

    # get first point (reference)
    self.STCm = self.dfDS.iloc[:,self.dfDS.columns.get_level_values(0).str.contains('STC')].values[0].tolist()
    self.STCm[1] += T0
    self.STCm[-1] = np.around(self.STCm[-1], 3)

    # get second point (complementary)
    self.NOCTm = self.dfDS.iloc[:,self.dfDS.columns.get_level_values(0).str.contains('NOCT')].values[0].tolist()
    self.NOCTm[1] += T0
    self.NOCTm[-1] = np.around(self.NOCTm[-1], 3)

    # get third point (complementary or regulatory)
    self.LICm = self.dfDS.iloc[:,self.dfDS.columns.get_level_values(0).str.contains('LIC')].values[0].tolist()
    self.LICm[1] += T0
    self.LICm[-1] = np.around(self.LICm[-1], 3)

    # get params of 11PFF model
    dfp      = dfs.iloc[:,dfs.columns.get_level_values(0).str.contains('Solution')]
    self.dfp = dfp.iloc[:,dfp.columns.get_level_values(0).str.contains('11PFF')]
    self.dfp = self.dfp.T.droplevel(0).droplevel(1).T

  def onclickMap(self, event):
    if event.button == MouseButton.LEFT:
      if event.xdata is not None and event.ydata is not None:
        self.user_T = float(event.xdata)
        self.user_S = float(event.ydata)
        if self.polygon.contains_point((event.xdata, event.ydata)):
          self.update_marker_select((event.xdata, event.ydata), '*')
        else:
          self.update_marker_select((event.xdata, event.ydata), '.')
      else:
        pass
      self.updateGraph()

  def update_marker_select(self, point, marker):
    # Remove the previous marker
    if self.last_marker:
      self.last_marker.remove()
    if self.last_vline:
      self.last_vline.remove()
    if self.last_hline:
      self.last_hline.remove()
    if self.last_annotation:
      self.last_annotation.remove()

    # Add a new marker
    self.last_marker, = self.axOptions.plot(point[0], point[1], marker=marker, color='red')
    
    # Add vertical and horizontal lines
    self.last_vline = self.axOptions.axvline(x=point[0], color='red', linestyle='--', linewidth=0.5)
    self.last_hline = self.axOptions.axhline(y=point[1], color='red', linestyle='--', linewidth=0.5)

    # Dynamic annotation positioning
    if point[0]<35:
      if point[1]<1000:
        text_x_offset = 15
        text_y_offset = 15
      else:
        text_x_offset = 10
        text_y_offset = -15
    else:
      if point[1]<1000:
        text_x_offset = -80
        text_y_offset = 15
      else:
        text_x_offset = -90
        text_y_offset = -25

    # Add an annotation close to the marker
    if not(text_x_offset is None):
      if not(text_y_offset is None):
        self.last_annotation = self.axOptions.annotate(f'({point[0]:.2f}, {point[1]:.2f})',
                                                      xy=point, xytext=(text_x_offset, text_y_offset),
                                                      textcoords='offset points',
                                                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    # Redraw the canvas
    self.canvasOptions.draw()

  @property
  def k11PFF(self):
    return ['b_ref', 'IL_ref', 'I0_ref', 'Rs_ref', 'Gp_ref', 
            'mI0', 'mRs', 'mGp', 
            'alphaT', 'deltaI0', 'deltaRs', 
            'T_ref', 'S_ref']
  
  @property
  def polygon_points(self):
    return[(25, 100), (15, 100), (15, 200), (15, 400), (15, 600),
                      (15, 800), (15, 1000), (25, 1100), (50, 1100), (75, 1100),
                      (75, 1000), (75, 800), (75, 600), (50, 200), (25, 100)]
  
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.activateWindow()
    sys.exit(app.exec())


