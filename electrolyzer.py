import numpy as np
import pandas as pd
import time
from vars import *
import random
import datetime
from math import log10, exp
from numpy import zeros
from scipy.optimize import fsolve
from helpers import vi_calc
from matplotlib.pyplot import plot, title, show
from datetime import timedelta
import math


class Electrolyzer():
    """
    Normalizes the input data by computing an online variance and mean
    """
    def __init__(self):
      
        ne = 2                                              # number moles of Water                                   
        Pe_out = 6000000                                   # (Pa) state of maximum charge in the tank, 60 bar
        charge_lvl_min = 0                                 # (%) minimum state of charge in the hydrogen tank
        charge_lvl_max = 99                                 # (%) maximum state of charge in the hydrogen tank
        soc_lvl_init = 1                                   # (%) initial state of charge
        
    
        # Compute starting parameters for model init
        self.min_charge = charge_lvl_min*1e-2*Pe_out             # min. state of charge in the hydrogen tank
        self.max_charge = charge_lvl_max*1e-2*Pe_out             # max. state of charge in the hydrogen tank
        soc_i = soc_lvl_init*1e-2*Pe_out                    # SOC initial
        self.n_i = soc_i*V_TANK/R/(T+273.15)
        
        DG_25 = 237000.0
        DG_80 = 228480.0
        DH = 286000.0
        DG = DG_25-(T-25)/55*(DG_25-DG_80)

        self.V_rev = DG/ne/F
        V_init = round(self.V_rev, 2)
        self.Vtn = DH/ne/F                                       # thermo-neutral voltage
        self.moles = 0

    def run(self, Pl):
        # Model parameters to be included in config.ini

        Pe = 101000                                         # (pa)
        
        eta_c = 0.8                                         # compressor's isentropic efficiency
        A = 1.25                                            # (m^2) area of electrode
        Ne = 100                                            # Number of Electrolyzers
        Nc = 100.0                                          # Number of cells connected in series
        
        # empiral parmeter from Ulleberg
        r1 = 7.331e-5                                       # (ohm m^2) ri parameter for ohmic resistance of electrolyte
        r2 = -1.107e-7                                      # (ohm m2 °C^-1)
        r3 = 0
        s1 = 1.586e-1                                       # (V) si and ti parameters for over-voltage on electrodes
        s2 = 1.378e-3                                       # (V°C^-1)
        s3 = -1.606e-5                                      # (V °C^-2)
        t1 = 1.599e-2                                       # (m^2 A^-1)
        t2 = -1.302                                         # (m^2 A^-1 C-1)
        t3 = 4.213e2                                        # (m^2 A^-1 C-2)

        # V vs i characteristic curve
        a1 = 0.995                                          # 99.5 %
        a2 = -9.5788                                        # (m ^ 2 * A ^ -1)
        a3 = -0.0555                                        # (m ^ 2 * A ^ -1 *°C)
        a4 = 0
        a5 = 1502.7083                                      # (m ^ 4 * A ^ -1)
        a6 = -70.8005                                       # (m ^ 4 * A ^ -1 *°C-1)
        a7 = 0
        gamma = 1.41
        cpH2 = 14.31                                        # Isobaric specific heat (Cp) kj / kg - K
        x0_1 = 1.6
        x0_2 = 80


        # Electrolyzer in action
        
        Pr = Pl/Ne                            # (W) to power one stack of electrolyzer
        V, Ir = fsolve(vi_calc, [x0_1, x0_2], args=(Pr, Nc, self.V_rev, T, r1, r2, s1, s2, s3, t1, t2, t3, A), maxfev=500)
        
        nf = a1*exp((a2+a3*T+a4*T**2)/(Ir/A)+(a5+a6*T+a7*T**2)/(Ir/A)**2)   # Compute Faraday Efficiency (Ulleberg)

        P = Ne*Nc*V*Ir

        # OUTPUT
        Qh2_m = Ne*Nc*Ir*nf/2/F                            # (mol/s)
        Qh2_m = Qh2_m * 60 * 60                            # (mol/h)
        m_dotH2 = Qh2_m*0.001                              # (kg/h)
        
        # Compressor model ERROR IN MOLES INDEX ERROR
        P_tank = self.moles*R*(T+273.15)/V_TANK                  # hydrogen tank


        
        #Tout = (T+273.15)*(P_tank/Pe)**((gamma-1)/gamma)
        #Wcomp = (m_dotH2/eta_c)*cpH2*(Tout-(T+273.15))       # Power for Compressor (kW)
        P_tot = P

        Wcomp = 0
        Tout = 0
       
        # Number of moles in time i in the tank
        self.moles += Qh2_m*1/Nt

        soc = P_tank
        #if soc >= self.max_charge:
            #print(self.moles*0.0101)
            #print('charged')

        return m_dotH2, Wcomp, P_tot, self.moles

    def reset(self):
        self.moles = 10

    def consume_moles_from_tank(self, moles):
        self.moles -= moles
    
    def get_moles(self):
        return self.moles

  


        
    

