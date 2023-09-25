from environment import EMS
from matplotlib import style
style.use('ggplot')
from vars import *
from itertools import count
import pickle as pkl
import os
import argparse
import torch
import pandas as pd
import numpy as np
from train_dqn import train_dqn
from train_ddpg import train_ddpg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='DDPG')
    parser.add_argument("--dynamic", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model_type", default='DDPG')
    parser.add_argument("--noisy", default=False, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(ckpt,model_name,dynamic,soft, eval, model_type, noisy):

    if not eval:
        if model_type == 'DQN':
            train_dqn(ckpt, model_name, dynamic, soft)
        elif model_type== 'DDPG':
            
            train_ddpg(ckpt, model_name, dynamic, noisy)
        elif model_type == 'Q':
            print("Q-LEARNING")
        else:
            print("No Model type selected")
            
    else:
        if ckpt:
            brain = torch.load(ckpt,map_location=torch.device('cpu'))
            brain.epsilon = 0
            brain.eps_end = 0
            brain.add_noise = False
            env = EMS(dynamic=True, eval=True)
            state = env.reset()    
            hydrogen_produced = [env.hydrogen]
            storage_state = [env.storage]
            prices = [env.price]
            power_from_grid = [env.power_from_grid]
            sun_power = [env.sun_power]
            wind_power = [env.wind_power]
            moles = [env.moles]
            natural_gas = [env.natural_gas]
            dates = [env.date]
            temperatures = [env.temperature]
            pv_generation = [env.pv_generation]
            natural_gas_prices = [env.natural_gas_price]
            gas_consumptions = [env.gas_consumption]
            electrolyzer = [0]
            battery_actions = [0]
            actions = [[0,0]]
            rewards = [0]
            print('Starting evaluation of the model')
            
            state = torch.tensor(state, dtype=torch.float).to(device)
            # Normalizing data using an online algo
            brain.normalizer.observe(state)
            state = brain.normalizer.normalize(state).unsqueeze(0) 
                  
            for t_episode in range(NUM_TIME_STEPS):
                action = brain.select_action(state).type(torch.FloatTensor)
                prices.append(env.price) # Will be replaced with environment price in price branch

                electrolyzer.append(action[0].numpy())
                battery_actions.append(action[1].numpy())

                
                actions.append(action.numpy())
                next_state, reward, done = env.step(action.numpy())
                rewards.append(reward)
                moles.append(env.moles)
                temperatures.append(env.temperature)
                pv_generation.append(env.pv_generation)
                natural_gas_prices.append(env.natural_gas_price)
                gas_consumptions.append(env.gas_consumption)
                hydrogen_produced.append(env.hydrogen)
                natural_gas.append(env.natural_gas)
                storage_state.append(env.storage)
                sun_power.append(env.sun_power)
                wind_power.append(env.wind_power)
                dates.append(env.date)
                power_from_grid.append(env.power_from_grid)
                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                    # normalize data using an online algo
                    brain.normalizer.observe(next_state)
                    next_state = brain.normalizer.normalize(next_state).unsqueeze(0)
                else:
                    next_state = None
                # Move to the next state
                state = next_state

            eval_data = pd.DataFrame()
            eval_data['PV Generation'] = pv_generation
            eval_data['Datetime'] = dates
            eval_data['Gas Consumption'] = gas_consumptions

            eval_data['Prices'] = prices
            eval_data['Prices Natural gas'] = natural_gas_prices
            eval_data['Moles'] = moles
            eval_data['Electrolyzer'] = battery_actions
            eval_data['Battery Action'] = electrolyzer
            eval_data['Actions'] = actions
            eval_data['Rewards'] = rewards
            eval_data['Temperatur'] = temperatures
            eval_data['Storage'] = storage_state
            eval_data['Power'] = power_from_grid
            eval_data['Sun Power'] = sun_power
            eval_data['Hydrogen'] = hydrogen_produced
            eval_data['Wind Power'] = wind_power
            eval_data['Natural Gas'] = natural_gas
            with open(os.getcwd() + '/data/output/' + model_type + '/' + model_name + '_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            print('Finished evaluation!')
            print('evaluating the policy...')

            temperatures = []
            battery_actions = []
            electrolyzer_action = []
            grid_prices = []
            ambient_temperatures = []
            battery_levels = [] 
            sun_powers = []
            times = []
            gas_prices = []
            actions = []

            #self.temperature, self.sun_power, self.price, self.storage, self.moles,  self.natural_gas_price

            for temperature in range(-10, 20, 1): 
                 
                for sun_power in np.arange(0,600,10):
                    for price in np.arange(-10,50,1):
                        for battery_level in np.arange(0, STORAGE_CAPACITY, STORAGE_CAPACITY/10):
                                for gas_price in np.arange(0,10,1):

                                    state = [temperature, sun_power, price, battery_level, gas_price] #, time
                                    state = torch.tensor(state, dtype=torch.float).to(device)
                                    state = brain.normalizer.normalize(state).unsqueeze(0)
                                    action = brain.select_action(state).type(torch.FloatTensor).numpy()
                                    

                                    
                                    electrolyzer_action.append(action[0])
                                    battery_actions.append(action[1])
                                    grid_prices.append(price)
                                    temperatures.append(temperature)
                                    actions.append(action)
                                    
                                    battery_levels.append(battery_level)
                                  
                                    sun_powers.append(sun_power)
                                    gas_prices.append(gas_price)

            eval_data = pd.DataFrame()
            eval_data['Temperature'] = temperatures
            eval_data['Actions'] = actions
            #eval_data['Grid Prices'] = grid_prices
            eval_data['Grid Prices'] = grid_prices
            eval_data['Battery Level'] = battery_levels
            eval_data['Battery Action'] = battery_actions
            eval_data['Electrolyzer Action'] = battery_actions
            eval_data['Sun Power'] = sun_powers
            eval_data['Natural Gas Price'] = gas_prices

            with open(os.getcwd() + '/data/output/' + model_type + '/' + model_name + '_policy_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

        else:
            print('If no training should be performed, then please choose a model that should be evaluated')

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))