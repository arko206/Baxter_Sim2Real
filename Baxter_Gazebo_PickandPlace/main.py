from env import Ur5
#from env2 import Ur5_vision
#from DDPG import DDPG
from TD3 import TD3
#from TD3_vision import TD3_vision
import numpy as np 
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import time
import argparse
import os
import numpy as np
import rospy
import actionlib
import time
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from tf import TransformListener
from math import pi
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys, tf
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from copy import deepcopy

# Define the base directory where you want to save log files
base_log_dir = "/home/robocupathome/arkaur5_ws/src/contexualaffordance/Ur5_DRL"

# Make sure the directory exists, if not, create it
if not os.path.exists(base_log_dir):
    os.makedirs(base_log_dir)



# File paths for logging rewards
train_reward_filename = os.path.join(base_log_dir, 'training_rewards.txt')
test_reward_filename = os.path.join(base_log_dir, 'testing_rewards.txt')



def get_time(start_time):
    m, s = divmod(int(time.time()-start_time), 60)
    h, m = divmod(m, 60)

    print ('Total time spent: %d:%02d:%02d' % (h, m, s))

def img_transform(img_memory, mode, frame_size=4):
    assert type(img_memory).__module__ == np.__name__, 'data type is not numpy'
    assert mode == 'img2txt' or mode == 'txt2img', 'Please use correct mode name 1.img2txt 2.txt2img'

    if mode == 'img2txt':
        size = img_memory.shape[0]
        img_memory = img_memory.reshape((size,-1))
    
    if mode == 'txt2img':
        size = img_memory.shape[0]
        h = w = np.sqrt(img_memory.shape[1] / (2 * frame_size))
        img_memory = img_memory.reshape((size,2,frame_size,h,w))
    
    return img_memory

def train(args, env, model):

    ###setting the saving path for the trained model
    if not os.path.exists(args.path_to_model+args.model_name+args.model_date):
        os.makedirs(args.path_to_model+args.model_name+args.model_date)

    #training reward list or cumulative reward list storing cumulative rewards during
    ### training
    
    #testing reward and steps list, which stores the rewards obtained and no. of steps
    ##during the evaluation process
    #test_reward_list, test_step_list = [], []
    #total_actor_loss_list, total_critic_loss_list = [], []
    start_time = time.time()
    if args.pre_train:
        #load pre_trained model 
        try:
            model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
            print ('load model successfully')
        except:
            print ('fail to load model, check the path of models')

        #print ('start random exploration for adding experience')

        ####Loading the TD3 Model
        #if args.model_name == 'TD3':
            #state = env.reset()

        #for step in range(args.random_exploration):
            ######Chnaged for Baxter's 7 joint angles
            #if args.model_name == 'TD3':
                #state_, action, reward, terminal = env.uniform_exploration(np.random.uniform(-1,1,7)*args.action_bound*7)
                #model.store_transition(state,action,reward,state_,terminal)
                #state = state_
                #if terminal:
                    #state = env.reset()
        #total_reward_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/reward.txt')
        #test_reward_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/test_reward.txt')
        #test_step_list = np.loadtxt(args.path_to_model+args.model_name+args.model_date_+'/test_step.txt')

    print ('start training')
    model.mode(mode='train')

    #training for observation with TD3 Algorithm

    train_episodes_list = []
    total_reward_list = []
    for epoch in range(args.train_epoch):

        ###creating an empty list for storing training episodes
        ### and cumulative reward in each episode
        

        if args.model_name == 'TD3':
            state = env.reset()
        total_reward = 0
        for i in range(args.train_step):
            if args.model_name == 'TD3':
                action = model.choose_action(state)
                state_, reward, terminal = env.step(action*args.action_bound)
                model.store_transition(state,action,reward,state_,terminal)
                state = state_
                total_reward = total_reward + (0.4)**i * reward
                if model.memory_counter > args.random_exploration:
                    model.Learn()
                if terminal:
                    state = env.reset()
        total_reward_list.append(total_reward)
        train_episodes_list.append(epoch)

            #print(total_reward_list)
            #print(train_steps_list)
                    
        #if len(total_actor_loss_list) == 0:
           #total_actor_loss = 'no actor loss evaluated'
        #else:
           #total_actor_loss = str(np.mean(total_actor_loss_list.detach().numpy()))
           ##tensor_list = [torch.tensor(value) for value in total_actor_loss_list]
           #stacked_tensor = torch.stack(tensor_list)
           #np_array = stacked_tensor.detach().numpy()
           #total_actor_loss = str(np.mean(np_array))
        
        #if len(total_critic_loss_list) == 0:
           #total_critc_loss = 'no critc loss evaluated'
        #else:
           #total_actor_loss = str(np.mean(total_actor_loss_list.detach().numpy()))
           #tensor_list_critc = [torch.tensor(value) for value in total_critic_loss_list]
           #stacked_tensor_critc = torch.stack(tensor_list_critc)
           #np_array_critc = stacked_tensor_critc.detach().numpy()
           #total_critc_loss = str(np.mean(np_array_critc))
        
        print ('epoch:', epoch,  '||',  'Reward:', total_reward)
        with open(train_reward_filename, "a") as file:
            file.write(f"epoch: {epoch} || Train_eward: {total_reward} \n")

        #begin testing and record the evalation metrics
        if (epoch+1) % args.test_epoch == 0:
            model.save_model(args.path_to_model+args.model_name, args.model_date+'/')

            ####Saving the Actor Critic Loss Plots for Training
            #model.plot_loss(args.path_to_model+args.model_name, args.model_date+'/', epoch)

            test_reward, epochs_step = test(args, env, model)
            #print(test_reward)
            #print(epochs_step)
            
            model.plot_loss(args.path_to_model+args.model_name, args.model_date+'/', epoch)
            
            model.mode(mode='train')
            print ('finish testing')

            # np.append(test_reward_list,avg_reward)
            #test_reward_list.append(avg_reward)

            # np.append(test_step_list,avg_step)
            #test_step_list.append(avg_step)

            plt.figure()
            plt.plot(epochs_step, test_reward, color='r')
            plt.ylabel('test_reward')
            plt.xlabel('testing epoch')
            plt.title(f'test_reward_trained(Epoch {epoch + 1})')
            plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/test_reward_trained_{epoch+1}.png')
            plt.close()
            #np.savetxt(args.path_to_model+args.model_name+args.model_date+f'/test_reward_{epoch+1}.txt',np.array(test_reward_list))

            #plt.figure()
            #plt.plot(np.arange(len(test_step_list)), test_step_list)
            #plt.ylabel('test_step')
            #plt.xlabel('training epoch / testing epoch')
            #plt.title(f'test_step_(Epoch {epoch + 1})')
            #plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/test_step_epoch_{epoch+1}.png')
            #plt.close()
            #np.savetxt(args.path_to_model+args.model_name+args.model_date+f'/test_step_{epoch+1}.txt',np.array(test_step_list))

            plt.figure()
            plt.plot(train_episodes_list[-10:], total_reward_list[-10:], color='g')
            plt.ylabel('Total_reward')
            plt.xlabel('training epoch')
            plt.title(f'Training_reward_(Epoch {epoch + 1})')
            plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/training_reward_{epoch+1}.png')
            plt.close()
            np.savetxt(args.path_to_model+args.model_name+args.model_date+f'/training_reward_{epoch+1}.txt',np.array(total_reward_list))
            get_time(start_time)
    plt.figure()
    plt.plot(train_episodes_list, total_reward_list, color='g')
    plt.ylabel('Total_reward')
    plt.xlabel('training epoch')
    plt.title('Training_reward_total')
    plt.savefig(args.path_to_model+args.model_name+args.model_date+f'/training_reward_{epoch+1}.png')
    plt.close()
            
    

def test(args,env, model):
    model.mode(mode='test')
    print ('start to test the model')
    try:
        model.load_model(args.path_to_model+args.model_name, args.model_date_+'/')
        #model.load_model('/home/robocupathome/arkaur5_ws/src/contexualaffordance/RL_trialTD3/06_06_2024')
        print(args.path_to_model+args.model_name, args.model_date_+'/')
        print ('load model successfully')
    except Exception as e:
        rospy.logwarn(e)
        print ('fail to load model, check the path of models')

    total_reward_list = []
    epochs_list = []
    #testing for vision observation
    for epoch in range(args.test_epoch):
        if args.model_name == 'TD3':
            state = env.reset()
        total_reward = 0
        for step in range(args.test_step):
            if args.model_name == 'TD3':
                action = model.choose_action(state,noise=None)
                state_, reward, terminal = env.step(action*args.action_bound)
                state = state_
                total_reward = total_reward + (0.4)**step * reward
                if terminal:
                    env.reset()
        total_reward_list.append(total_reward)
        print ('testing_epoch:', epoch,  '||',  'Reward:', total_reward)

        ###Writing in file
        with open(test_reward_filename, "a") as file:
            file.write(f"epoch: {epoch} || Test_Reward: {total_reward}\n")

        ####adding each iterations
        epochs_list.append(epoch)


    #average_reward = np.mean(np.array(total_reward_list))
    #average_step = 0 if steps_list == [] else np.mean(np.array(steps_list))

    return total_reward_list, epochs_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #select env to be used
    parser.add_argument('--env_name', default='empty')
    #select model to be used
    parser.add_argument('--model_name', default='TD3')
    #Folder name saved as date
    parser.add_argument('--model_date', default='/24_06_2024')
    #Folder stored with trained model weights, which are used for transfer learning
    parser.add_argument('--model_date_', default='/24_06_2024')
    parser.add_argument('--pre_train', default=False)

    #####has to be changed
    parser.add_argument('--path_to_model', default='/home/robocupathome/rosur5_ws/src/contexualaffordance/DRLMODELTD3')
    #The maximum action limit

    ####has to be chnaged
    parser.add_argument('--action_bound', default=np.pi/72, type=float) #pi/36 for reachings
    parser.add_argument('--train_epoch', default=200, type=int)
    parser.add_argument('--train_step', default=200, type=int)
    parser.add_argument('--test_epoch', default=10, type=int)
    parser.add_argument('--test_step', default=200, type=int)
    #exploration (randome action generation) steps before updating the model
    parser.add_argument('--random_exploration', default=1000, type=int)
    #store the model weights and plots after epoch number
    parser.add_argument('--epoch_store', default=10, type=int)
    #Wether to use GPU
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    #assert args.env_name == 'empty' or 'vision', 'env name: 1.empty 2.vision'
    assert args.env_name == 'empty', 'env name: 1.empty'
    if args.env_name == 'empty': env = Ur5()
    #if args.env_name == 'vision': env = Ur5_vision()

    #assert args.model_name == 'TD3_vision' or 'TD3' or 'DDPG', 'model name: 1.TD3_vision 2.TD3 3.DDPG'
    assert args.model_name == 'TD3', 'model name: 1.TD3'
    #if args.model_name == 'TD3_vision': model = TD3_vision(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    if args.model_name == 'TD3': model = TD3(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)
    #if args.model_name == 'DDPG': model = DDPG(a_dim=env.action_dim,s_dim=env.state_dim,cuda=args.cuda)

    assert args.mode == 'test' or 'test', 'mode: 1.train 2.test'
    if args.mode == 'train': 
        train(args, env, model)

    if args.mode == 'test': 
        env.duration = 0.1
        test(args, env, model)
