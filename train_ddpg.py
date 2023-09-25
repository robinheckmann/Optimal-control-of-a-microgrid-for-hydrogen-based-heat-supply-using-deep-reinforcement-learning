from DDPG import DDPGagent
from environment import EMS
from matplotlib import style
from vars import *
from itertools import count
import pickle as pkl
import os
import sys
import torch



def train_ddpg(ckpt, model_name, dynamic, noisy, save_best = True):
    env = EMS(dynamic)
    total_rewards = []
    
    if ckpt:
        brain = torch.load(ckpt,map_location=torch.device('cpu'))
    else:
        brain = DDPGagent(mem_size=MEMORY_SIZE, add_noise = noisy)



    for i_episode in range(NUM_EPISODES):
        # Initialize the environment.rst and state
        brain.reset()
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).to(device)

        # Normalizing data using an online algo      
        brain.normalizer.observe(state)
        state = brain.normalizer.normalize(state).unsqueeze(0)
        episode_reward = 0

        for t in count():
            # Select and perform an action
            action = brain.select_action(state).type(torch.FloatTensor)
            next_state, reward, done = env.step(action.numpy())
            episode_reward += reward
            reward = torch.tensor([reward], dtype=torch.float, device=device)

            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                # normalize data using an online algo
                brain.normalizer.observe(next_state)
                next_state = brain.normalizer.normalize(next_state).unsqueeze(0)
            else:
                next_state = None

            # Insert Tupel (a,a,s+1,r) to replay buffer
            brain.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            brain.optimize_model()

            if done:
                total_rewards.append(episode_reward)
                break
        

        # save model every 10 episodes
        if i_episode%10 == 0:
            #torch.save(brain, os.getcwd()  + '/' + model_name + '.pt')
            torch.save(brain, os.getcwd()  + '/data/output/DDPG/models/' + model_name + '.pt')
            with open(os.getcwd() + '/data/output/DDPG/' + model_name + '_dynamic_' + str(dynamic) + '_noise_' + str(noisy) +  '_rewards_dqn.pkl', 'wb') as f:
                pkl.dump(total_rewards, f)

        sys.stdout.write('Finished episode {} with reward {}\n'.format(i_episode, round(episode_reward)))


    model_params = {
        'NUM_EPISODES': NUM_EPISODES,
        'EPSILON': EPSILON,
        'EPS_DECAY': EPS_DECAY,
        'LEARNING_RATE_ACTOR':LEARNING_RATE_ACTOR,
        'LEARNING_RATE_CRITIC': LEARNING_RATE_CRITIC,
        'GAMMA': GAMMA,
        'TARGET_UPDATE': TARGET_UPDATE,
        'BATCH_SIZE': BATCH_SIZE,
        'MEMORY_SIZE':MEMORY_SIZE,
        'ETA_BATTERY':ETA
    }

    total_rewards.append(model_params)
    with open(os.getcwd() + '/data/output/DDPG/' + model_name + '_dynamic_' + str(dynamic) + '_rewards_dqn.pkl', 'wb') as f:
        pkl.dump(total_rewards, f)

    # Saving the final model
    torch.save(brain, os.getcwd()  + '/data/output/DDPG/' + model_name + '.pt')
    print('Complete')