import numpy as np
import pandas as pd
import time
from vars import *
import random
import datetime
from torch.utils.tensorboard import SummaryWriter

from math import log10, exp
from numpy import zeros
from scipy.optimize import fsolve
from matplotlib.pyplot import plot, title, show
from electrolyzer import Electrolyzer
from datetime import timedelta
from datetime import datetime

writer = SummaryWriter()


class EMS:
    def __init__(self, dynamic=False, eval=False, pv_gen=True, wind_gen=False):
                
        self.eval = eval
        self.dynamic = dynamic
        self.pv_gen = pv_gen
        self.wind_gen = wind_gen

        self.electrolyzer = Electrolyzer()

        self.electrolyzer.reset()
        self.moles = self.electrolyzer.get_moles()
        
        if self.eval:
            #self.random_day = random.randint(43825, 52608 - NUM_TIME_STEPS)
            self.random_day = 37946
        else:
            self.random_day = random.randint(0, 37945 - NUM_TIME_STEPS)
            #self.random_day = 37945 - 8760


        self.sun_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,1]
        self.sun_power = self.sun_powers[self.random_day]

        self.wind_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,7]
        self.wind_power = self.wind_powers[self.random_day]

        self.temperatures = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,6]
        self.temperature = self.temperatures[self.random_day]

        self.prices = pd.read_csv('data/environment/prices/data.csv', header=0, delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,5]
        self.price = self.prices[self.random_day]

        self.dates = pd.read_csv('data/environment/prices/data.csv', header=0, delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,0]
        self.date = self.dates[self.random_day]

        

        

        # reset parameters
        random.seed(time.perf_counter())
        self.natural_gas_price = 0
        self.natural_gas = 0                    # (kWh/kg)
        self.temperature = 0
        self.storage = 0
        self.hydrogen = 0
        self.power_from_grid = 0
        self.done = False
        self.time = 0
        self.gas_consumption = 0
        self.pv_generation = 0
 

    def step(self, action):

        # produce hydrogen with electrolyzer

        # Photovoltaik
        if self.pv_gen:
            #pv_generation = np.minimum(self.sun_power * PV_EFFICIENCY * PV_SURFACE, PV_CAPACITY)     
            pv_generation = self.sun_power
        else:
            pv_generation = 0 

        self.pv_generation = pv_generation
        # BATTERY STORAGE
        if action[1] >= 0:        
            Pbattery = np.minimum(action[1] * C_MAX, (STORAGE_CAPACITY - self.storage) / ETA)
            self.storage += int(Pbattery * ETA)
        else:           
            Pbattery = np.maximum(action[1] * D_MAX, (MIN_STORAGE - self.storage) * ETA)
            self.storage += int(Pbattery / ETA)
        
      
        

    


        
        #Pel = action[0] * pv_generation + abs(np.minimum(Pbattery,0))
        #Pel = action[0] * (ELECTROLYZER_POWER * 0.1 + pv_generation + np.maximum(0,Pbattery))
        Pel = action[0] * ELECTROLYZER_POWER 
        electrolyser_output, Wcomp, P_tot, moles = self.electrolyzer.run(Pel)
        self.hydrogen = electrolyser_output
        self.moles = self.electrolyzer.get_moles()                          # store hydreogen

        # household gas consumption modelling
        a = 2.794
        b = -37.2
        c = 5.40
        d = 0.171391
        h = 16*150/365/24                                                   # 16 m3 Gas pro Quadratmeter im Jahr (m3)
        h = h * 10                                                          # 10 kWh pro m3 Heizwert (L/H-Gas) (kWh/kg)
        gas_consumption = a/(1+((b/(self.temperature-40))**c))+d
        gas_consumption = gas_consumption * h * AMOUNT_HOMES
        self.gas_consumption = gas_consumption        
        
        # calculate hydrogen & gas usage
        hydrogen_power = np.minimum(gas_consumption, self.moles * 0.0101 * 33.3)     
        self.electrolyzer.consume_moles_from_tank(hydrogen_power/0.0101/33.3)       # discharge action 
        natural_gas_needed = np.maximum(0, gas_consumption - hydrogen_power)        # replace missing green hydrogen with grey hydrogen ()
        self.natural_gas = natural_gas_needed / 10                                  # (m3) Gas
     
     


      
        
        # wind turbines
        if self.wind_gen:
            wind_speed = np.maximum(round(self.wind_power), 25)
            cp = [0.0,0.0,0.28,0.37,0.41,0.44,0.45,0.45,0.45,0.43,0.40,0.35,0.3,0.24,0.20,0.16,0.13,0.11,0.09,0.08,0.07,0.06,0.05,0.04,0.04,0.03]
            wind_generation = 0.5*1.225*self.wind_power**3*(63.5**2)*cp[wind_speed] * NUM_WINDTURBINES         
        else:
            wind_generation = 0
        

        self.power_from_grid = Pel + Pbattery - pv_generation - wind_generation 

        self.power_from_grid *= 1e-6  

        # calculate rewards
        r = self.reward(self.power_from_grid)
        
        # tensorboard scalars
        writer.add_scalar('Consumption/Temperature', self.temperature, self.time)
        writer.add_scalar('States/Hydrogen Storage', self.moles * 0.0101, self.time)    
        writer.add_scalar('States/Price', self.price, self.time)
        writer.add_scalar('States/External Energy Source/Power From Grid', self.power_from_grid, self.time)
        writer.add_scalar('States/External Energy Source/Natural Gas Consumption', self.natural_gas, self.time)
        writer.add_scalar('States/External Energy Source/Natural Gas Price', self.natural_gas_price, self.time)
        writer.add_scalar('States/External Energy Source/Wind Generation', wind_generation, self.time)
        writer.add_scalar('Actions/PV Generation', pv_generation * 1e-6, self.time)
        writer.add_scalar('Actions/Electrolyzer', action[0], self.time)
        writer.add_scalar('Actions/Storage', self.storage, self.time)
        
      
        self.time +=1
        if self.dynamic:
            self.sun_power = self.sun_powers[self.random_day + self.time]
            self.sun_power = np.maximum(0, self.sun_power)
            self.wind_power = self.wind_powers[self.random_day + self.time]
            self.price = self.prices[self.random_day + self.time]
            self.temperature = self.temperatures[self.random_day + self.time]
            self.date = self.dates[self.random_day + self.time]
            date = datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S')
            date = date.strftime('%Y-%m-%d')
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == date];
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=1)).strftime("%Y-%m-%d")];
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=2)).strftime("%Y-%m-%d")];       
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=3)).strftime("%Y-%m-%d")];
            if row.empty:
                row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=4)).strftime("%Y-%m-%d")];              
                      
            self.natural_gas_price = row['Price'].iloc[0]

        if self.time >= NUM_TIME_STEPS:
            self.done = True

        return [self.temperature, self.sun_power, self.price, self.storage, self.natural_gas_price], r, self.done 


    def reward(self, P_grid):
        
        if P_grid >=0:
            paid_price = - P_grid*self.price           # buy mal 2
        else:
            paid_price = P_grid*self.price * 1.5   # sell 1.5
            
        
        price_natural_gas = self.natural_gas / 28.32 * self.natural_gas_price 

        reward = paid_price - price_natural_gas + self.hydrogen



        return reward

    def reset(self):
      
        if self.eval:
            self.random_day = 37946
        else:
            self.random_day = random.randint(0, 37945 - NUM_TIME_STEPS)

        ## Resetting parameters

        
        self.sun_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,1]
        self.sun_power = self.sun_powers[self.random_day]

        self.wind_powers = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,7]
        self.wind_power = self.wind_powers[self.random_day]

        self.temperatures = pd.read_csv('data/environment/renewables/data.csv', header=0, delimiter=',').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,6]
        self.temperature = self.temperatures[self.random_day]

        self.prices = pd.read_csv('data/environment/prices/data.csv', header=0,delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,5]
        self.price = self.prices[self.random_day]

        self.dates = pd.read_csv('data/environment/prices/data.csv', header=0, delimiter=';').iloc[self.random_day:self.random_day+NUM_TIME_STEPS+1,0]
        self.date = self.dates[self.random_day]
        date = datetime.strptime(self.date, '%Y-%m-%d %H:%M:%S')
        date = date.strftime('%Y-%m-%d')
     
        self.natural_gas_prices = pd.read_csv('data/environment/gas/data.csv', header=0, delimiter=';')
        self.natural_gas_prices['Date'] = pd.to_datetime(self.natural_gas_prices['Date'])  
      
        row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == date];
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=1)).strftime("%Y-%m-%d")];
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=2)).strftime("%Y-%m-%d")];       
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=3)).strftime("%Y-%m-%d")];
        if row.empty:
            row = self.natural_gas_prices.loc[self.natural_gas_prices['Date'] == (pd.to_datetime(date) - timedelta(days=4)).strftime("%Y-%m-%d")];
       
        
        self.natural_gas_price = row['Price'].iloc[0]
        
        self.storage = 0
        self.pv_generation = 0

        self.natural_gas = 0
        self.time = 0
        self.hydrogen = 0
        self.gas_consumption = 0
        self.done = False
        self.soc = 0
        self.electrolyzer.reset()
        self.moles = self.electrolyzer.get_moles()

        return [self.temperature, self.sun_power, self.price, self.storage, self.natural_gas_price] 
