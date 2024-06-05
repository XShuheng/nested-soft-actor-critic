"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import random
import time
import sys
from pypower.api import case30, ppoption, runpf, case30
from case31 import case31


class ENV1():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 20  # vlotege 发电机功率，######load shedding
        self.observation_space =66#有功无功负荷

class IEGS():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 25  # vlotege 发电机功率，######load shedding
        self.observation_space =43#有功无功负荷

    def reset(self):
        maxPl=np.zeros(shape=(42,25))
        minPl = np.zeros(shape=(42, 25))
        maxPl1 = np.zeros(shape=(42, 25))
        minPl1 = np.ones(shape=(42, 25))
        rate= np.array([
             1, 1.041666667, 0.912, 0.85, 0.833333333, 0.733333333, 0.75, 1.083333333, 1.466666667, 1.883333333, 1.966666667, 2.216666667, 2.066666667, 1.941666667, 2.116666667,
        2.233333333, 2.316666667, 2.208333333, 2, 2, 1.833333333, 1.633333333, 1.433333333,1,1
        ])#每小时负荷比例
        parameters1 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*1.5
        parameters2 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*0.5
        #负荷节点上下限度
        data=np.array([2,3,4,7,8,10,11,12,14,15,16,17,18,19,20,21,23,24,26,29,30])-1 #负荷节点
        for i in range(25):
            maxPl[:, i] = rate[i] * parameters1
            minPl[:, i] = rate[i] * parameters2
        load_shedding=maxPl-minPl
        state = np.random.uniform(maxPl, minPl)
        state1 = np.random.uniform(maxPl1, minPl1)
        #state1=state1*(maxPl-minPl)
        max_action = np.array(
            [1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 160, 100, 110, 60, 80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1])
        #max_action = np.array([1.05,1.05,1.05,1.05,1.05,1.05,160,100,110,60,80,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])#前六个是发电机电压控制，7-11是PV发电机输出功率控制，后面1是PQ节点的load shedding
        #min_action = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        min_action = np.array(
            [1.05, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #min_action = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0])
        #max_action=np.array([2,60])
        #min_action=np.array([1,30])
        action_data=max_action-min_action
        action_basic=min_action
        #load_shedding=[]

        #data=GasData()
        return state, action_data, data, action_basic,load_shedding,minPl

    def step(self, action, action_data,data,action_basic,state_,state,step,load_shedding,attack,minPl,mark):
        #state_shedding=np.zeros(shape=(42))
        #state=state*load_shedding[:,step]+minPl[:,step]
        #state_=state_*load_shedding[:,step+1]+minPl[:,step+1]
        s_ = state_
        action = (action + 10) / 20
        mark1=np.array(
            [5,7,9,10,11,12,13,14,15,16,17,18,19,20])-1
        action = action * action_data + action_basic
        action1=action
        j=0
        action = np.zeros(shape=(32))
        for i in range(11):
            action[i] = action1[i]
        for i in range(len(mark1)):
            action[i+11]=action1[j+11]
            j+=1



        if attack!=-1:


            attack = (attack + 10) / 20*20
            for i in range(len(attack)):
                attack[i]=round(attack[i])

                if attack[0]==0:
                    attack[0]=7
                elif attack[0]==1:
                    attack[0]=8
                elif attack[0]==2:
                    attack[0]=9
                elif attack[0]==3:
                    attack[0]=12
                elif attack[0]==4:
                    attack[0]=16
                elif attack[0]==5:
                    attack[0]=17
                elif attack[0]==6:
                    attack[0]=18
                elif attack[0]==7:
                    attack[0]=19
                elif attack[0]==8:
                    attack[0]=20
                elif attack[0]==9:
                    attack[0]=21
                elif attack[0]==10:
                    attack[0]=22
                elif attack[0]==11:
                    attack[0]=23
                elif attack[0]==12:
                    attack[0]=24
                elif attack[0]==13:
                    attack[0]=25
                elif attack[0]==14:
                    attack[0]=26
                elif attack[0]==15:
                    attack[0]=28
                elif attack[0]==16:
                    attack[0]=31
                elif attack[0]==17:
                    attack[0]=33
                elif attack[0]==18:
                    attack[0]=37
                elif attack[0]==19:
                    attack[0]=38
                elif attack[0]==20:
                    attack[0]=39



                #if attack[0]==0:
                #    attack[0]=7
                #if attack[0]==1:
                #    attack[0]=8
                #if attack[0]==2:
                #    attack[0]=9
                #if attack[0]==3:
                #    attack[0]=12
                #if attack[0]==4:
                #    attack[0]=16
                #if attack[0]==5:
                #    attack[0]=17
                #if attack[0]==6:
                #    attack[0]=18
                #if attack[0]==7:
                #   attack[0]=19
                #if attack[0]==8:
                #    attack[0]=20
                #if attack[0]==9:
                #    attack[0]=21
                #if attack[0]==10:
                #    attack[0]=22
                #if attack[0]==11:
                #    attack[0]=23
                #if attack[0]==12:
                #    attack[0]=24
                #if attack[0]==13:
                #    attack[0]=25
                #if attack[0]==14:
                #    attack[0]=26
                #if attack[0]==15:
                #    attack[0]=28
                #if attack[0]==16:
                #    attack[0]=31
                #if attack[0]==17:
                #    attack[0]=33
                #if attack[0]==18:
                #    attack[0]=37
                #if attack[0]==19:
                #    attack[0]=38
                #if attack[0]==20:
                #    attack[0]=39

                #if attack[i]==5:
                #    attack[i]=7

                ####################
                #if attack[i]==0:
                #    attack[i]=13
                #if attack[i]==29:
                #    attack[i]=30
                #if attack[i]==15:
                #    attack[i]=14
                #if attack[i]==35:
                #    attack[i]=12
                #if attack[i]==10:
                #    attack[i]=33
                #if attack[i]==40:
                #    attack[i]=12
                #if attack[i]==34:
                #    attack[i]=33
###############################################

            #for N-K attack
            ###################for i in range(len(attack)):
            #    mark=Find(attack[i],attack,i)
            #    if mark!=[]:
            #        p=1
            #        for j in range(len(mark)):
            #            attack[mark[j]]=attack[mark[j]]-p
            #########################            p+=1
        c=0
        c1=0

        load_shedding_s = 0
        penalty = 0
        shedding = np.zeros(shape=(21))

        action[11]+=0.00001
        action[12] += 0.00001
        #if load_shedding!=[]:
        #    for i in range(21):
        #        shedding[i]=load_shedding[i,step]*action[11]
        #    state_shedding[0:21] = state[0:21] - shedding
        #    state_shedding[22:42] = state[22:42]
        #    powerflow2 = runpf(case30(action, state_shedding, data, attack,0,0,0,[]))

        powerflow= runpf(case30(action, state, data,[],0,0,0,[]))

        load_shed=0
        for i in range(len(data)):
            load_shed+=action[i+11]*state[i]

        if powerflow[0]["success"]==False :
            reward=-1e4
            reward2= np.random.uniform(100, 99.99)
            record = reward, c1, action, attack
            reward2_t=reward2
            reward11t=reward
            reward00t = reward
        else:
            if attack!=-1:
                if attack[0]==12 or attack[0]==33:
                    a=1
                powerflow2 = runpf(case30(action, state, data, attack,[],[],[],[]))

                for i in range(1):
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 1] - powerflow2[0]["gen"][i, 8])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 9] - powerflow2[0]["gen"][i, 1]))) / (
                                       (powerflow2[0]["gen"][i, 8]) - powerflow2[0]["gen"][i, 9])) ** 2
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / (
                                       (powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c1 += ((SolveMax(0,
                                     (powerflow2[0]["gen"][i + 1, 2] - powerflow2[0]["gen"][i + 1, 3])) + SolveMax(
                        0, (
                                powerflow2[0]["gen"][i + 1, 4] - powerflow2[0]["gen"][i + 1, 2]))) / (
                                   (powerflow2[0]["gen"][i + 1, 3]) - powerflow2[0]["gen"][i + 1, 4])) ** 2
                for i in range(30):
                    c1 += ((SolveMax(0, powerflow2[0]["bus"][i, 7] - 1.05) + SolveMax(0,
                                                                                      0.95 - powerflow2[0]["bus"][
                                                                                          i, 7])) / 0.1) ** 2
                # c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c1 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c1 = (c1 / 71) ** 0.5
                sou=int(powerflow2[0]["branch"][int(attack[0]), 0])-1
                load_s = powerflow2[0]["bus"][sou, 2]
                end=int(powerflow2[0]["branch"][int(attack[0]), 1])-1
                load_e = powerflow2[0]["bus"][end, 2]
                load_shedding_s = 0
                load_shedding_e = 0
                load_ss1 = 0
                load_ss2 = 0
                load_ee1 = 0
                load_ee2 = 0
                load_ss = 0
                load_ee = 0
                if sou==26 or end==26 or attack[0] == 21 or attack[0] == 24 or attack[0] == 22 or attack[0] == 23 or attack[0] == 18 or attack[0] == 25:
                    if attack[0] == 21 or attack[0] == 24 or attack[0] == 22 or attack[0] == 23 or attack[0] == 18 or attack[0] == 25:
                        if attack[0] == 21 or attack[0] == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 0))
                        if attack[0] == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 1))
                        if attack[0] == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 2))
                        if attack[0] == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 3))
                        if attack[0] == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 4))
                    if sou == 26 or end == 26:
                        powerflow2 = runpf(case30(action, state, data, attack, powerflow2[0]["bus"][28, 2]/2, powerflow2[0]["bus"][29, 2]/2, 28, 29))
                        load_shedding_s = powerflow2[0]["bus"][28, 2]/2
                        load_shedding_e = powerflow2[0]["bus"][29, 2]/2
                else:
                    for i in range(41):
                        mark1=Find(sou+1,powerflow2[0]["branch"][:, 0],int(attack[0]))#以sou为入口
                        mark2=Find(end+1,powerflow2[0]["branch"][:, 1],int(attack[0]))#以end为出口
                        mark11 = Find(sou+1, powerflow2[0]["branch"][:, 1], int(attack[0]))#以sou为出口
                        mark22 = Find(end+1, powerflow2[0]["branch"][:, 0], int(attack[0]))#以end为入口
                    if mark1!=[]:
                        for i in range(len(mark1)):
                            if powerflow2[0]["branch"][mark1[i], 5]<-powerflow2[0]["branch"][mark1[i], 13]:
                                load_ss1= 0.9*(sum(powerflow2[0]["branch"][mark1[j], 5] for j in range(len(mark1))))
                            else:
                                load_ss1 =0
                    if mark22 != []:
                        for i in range(len(mark22)):
                            if powerflow2[0]["branch"][mark22[i], 5]<-powerflow2[0]["branch"][mark22[i], 15]:
                                load_ee1= 0.9*(sum(powerflow2[0]["branch"][mark22[j], 5] for j in range(len(mark22))))
                            else:
                                load_ee1 =0
                    if mark11 != []:
                        for i in range(len(mark11)):
                            if powerflow2[0]["branch"][mark11[i], 5]<-powerflow2[0]["branch"][mark11[i], 15]:
                                load_ss2= 0.9*(sum(powerflow2[0]["branch"][mark11[j], 5] for j in range(len(mark11))))
                            else:
                                load_ss2 =0
                    if mark2 != []:
                        for i in range(len(mark2)):
                            if powerflow2[0]["branch"][mark2[i], 5]<-powerflow2[0]["branch"][mark2[i], 13]:
                                load_ee2= 0.9*(sum(powerflow2[0]["branch"][mark2[j], 5] for j in range(len(mark2))))
                            else:
                                load_ee2 =0
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss1 =0
                        load_ss2 = 0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee1 = 0
                        load_ee2 = 0
                    if load_ss1!=0 or load_ss2!=0 or load_ee1!=0 or load_ee2!=0:
                        if mark1!=[]:
                            load_ss += load_ss1/ len(mark1)
                        if mark11 != []:
                            load_ss += load_ss2 / len(mark11)
                        if mark22!=[]:
                            load_ee += load_ee1/ len(mark22)
                        if mark2 != []:
                            load_ee += load_ee2 / len(mark2)

                        #load_ss=(load_ss1+load_ss2)/(len(mark1)+len(mark11))
                        #load_ee = (load_ee1 + load_ee2) / (len(mark2) + len(mark22))
                    else:
                        load_ss =load_s
                        load_ee=load_e
                    #if load_ss==0:
                    #    load_ss=0.8*load_s
                    #if load_ee==0:
                    #    load_ee=0.8*load_e
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss =0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee = 0
                    if powerflow2[0]["success"] == True:

                        load_shedding_s = load_s - load_ss
                        load_shedding_e = load_e - load_ee
                        if load_shedding_s < 0:
                            if sou!=1 and sou!=23 and sou!=0 and sou!=22:
                                penalty=-load_shedding_s
                                load_ss= load_s
                                load_shedding_s = 0
                            if sou == 1:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][6, 2] + 0.2 * powerflow2[0]["bus"][7, 2]
                                powerflow2 = runpf(case30(action, state, data, attack, 0.8 * powerflow2[0]["bus"][6, 2], 0.8 * powerflow2[0]["bus"][7, 2], 6, 7))
                            elif sou == 23:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][21, 2]
                                powerflow2 = runpf(case30(action, state, data, attack, 0.2 * powerflow2[0]["bus"][21, 2], 0, 21, -1))
                            elif sou == 22:
                                #action1[9] = (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_s = 0
                                penalty+=(action[9]-(load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]))
                                action[9]= (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9])
                                powerflow2 = runpf(case30(action, state, data, attack, load_ss, load_ee, sou, end))
                            elif sou == 0:
                                #action1[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]) / action_data[6] * 20 - 10

                                load_shedding_s = 0
                                penalty +=(action[6]-(load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]))
                                action[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6])
                                powerflow2 = runpf(case30(action, state, data, attack, load_ss, load_ee, sou, end))
                            else:
                                load_shedding_s = 0
                                powerflow2 = runpf(case30(action, state, data, attack, load_ss, load_ee, sou, end))
                        elif load_shedding_e < 0:
                            if end != 22:
                                penalty=-load_shedding_e
                                load_ee = load_e
                                load_shedding_e=0
                            if end == 22:
                                #action1[9] = (load_e + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_e = 0
                                penalty += (action[9] - (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]))
                                action[9] = (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9])
                                powerflow2 = runpf(case30(action, state, data, attack, load_ss, load_ee, sou, end))
                            else:
                                load_shedding_e = 0
                                powerflow2 = runpf(case30(action, state, data, attack, load_ss, load_ee, sou, end))
                        else:
                            if attack==12 or attack==33:
                                a=1
                            else:
                                powerflow2 = runpf(case30(action, state, data, attack, load_ss, load_ee, sou, end))

                            # if end==26:
                            #    action1[8]=(load_e+powerflow2[0]["branch"][mark2[0], 5]-action_basic)/action_data*20-10
                            #a=1
                            #if load_ss>load_s:
                            #    load_shedding_s=0
                            #if load_ee>load_e:
                            #    load_shedding_e=0

                    if attack[0] == 21 or attack[0] == 24 or attack[0] == 22 or attack[0] == 23 or attack[0] == 18 or attack[0] == 25:
                        if load_s - load_ss<0:
                            a=1
                        if load_e - load_ee<0:
                            a=1
                        if attack[0] == 21 or attack[0] == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 0))
                        if attack[0] == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 1))
                        if attack[0] == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 2))
                        if attack[0] == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 3))
                        if attack[0] == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case30(action, state, data, attack, 0, 0, 0, 4))
                if attack[0]==12 :
                    load_shedding_s=powerflow[0]["bus"][10, 2]
                    load_shedding_e = 0
                    penalty=0
                if attack[0]==33 :
                    load_shedding_s=1
                    load_shedding_e = powerflow[0]["bus"][25, 2]
                    penalty=0
            for i in range(1):#slack节点有功功率和无功功率是否超出限额
                c+=((SolveMax(0,(powerflow[0]["gen"][i, 1]-powerflow[0]["gen"][i, 8]))+SolveMax(0,(powerflow[0]["gen"][i, 9]-powerflow[0]["gen"][i, 1])))/((powerflow[0]["gen"][i, 8])-powerflow[0]["gen"][i, 9]))**2
                c += ((SolveMax(0, (powerflow[0]["gen"][i, 2] - powerflow[0]["gen"][i, 3])) + SolveMax(0, (powerflow[0]["gen"][i, 4] - powerflow[0]["gen"][i, 2]))) / ((powerflow[0]["gen"][i, 3]) - powerflow[0]["gen"][i, 4])) ** 2
            for i in range(5):#PV节点无功功率是否超出限额
                c += ((SolveMax(0, (powerflow[0]["gen"][i+1, 2] - powerflow[0]["gen"][i+1, 3])) + SolveMax(0, (
                            powerflow[0]["gen"][i+1, 4] - powerflow[0]["gen"][i+1, 2]))) / (
                                  (powerflow[0]["gen"][i+1, 3]) - powerflow[0]["gen"][i+1, 4])) ** 2
            for i in range(30):#节点电压是否超出限额
                c+=((SolveMax(0,powerflow[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow[0]["bus"][i,7]))/0.1)**2
            #c=(c/31)**0.5
            for i in range(41):#线路流量是否超出限额
                c+=(SolveMax(0,powerflow[0]["branch"][i,13]-powerflow[0]["branch"][i,5])/powerflow[0]["branch"][i,5])**2
            c = (c / 72) ** 0.5


            ###
            c2=0

            if attack!=-1:
                for i in range(1):
                    c2+=((SolveMax(0,(powerflow2[0]["gen"][i, 1]-powerflow2[0]["gen"][i, 8]))+SolveMax(0,(powerflow2[0]["gen"][i, 9]-powerflow2[0]["gen"][i, 1])))/((powerflow2[0]["gen"][i, 8])-powerflow2[0]["gen"][i, 9]))**2
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / ((powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i+1, 2] - powerflow2[0]["gen"][i+1, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i+1, 4] - powerflow2[0]["gen"][i+1, 2]))) / (
                                      (powerflow2[0]["gen"][i+1, 3]) - powerflow2[0]["gen"][i+1, 4])) ** 2
                for i in range(30):
                    c2+=((SolveMax(0,powerflow2[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow2[0]["bus"][i,7]))/0.1)**2
                #c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c2 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c2 = (c2 / 71) ** 0.5
            if c1==0:
                load_shedding_s+=c2*1e5

            ###


            if powerflow[0]["success"]==False :
                reward0=-1e8
                reward00 = -1e8
            else:
                reward00 = -(sum(powerflow[0]["gen"][:, 1])+c*1e3)
                reward0 = -(sum(powerflow[0]["gen"][:, 1]) )#R0
            aaa = np.random.uniform(1000, 999.99)
            if attack==-1:
                reward=reward00
                reward2=0
                reward1=[]
                record = reward, reward0, reward1, c, action
                reward00t=0
                reward2_t=0
                reward11t=0
            else:
                if powerflow2[0]["success"]==False :
                    if mark==1:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e4 + (load_shed) * 5 + 2000)
                    else:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 5 + 2000)
                    reward11t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 5 + 2000)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 )
                    #reward1=-1e6
                    #reward11 = -1e6

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1])  + (load_shed) * 1e2)
                    reward2 =1000
                    reward2_t =20
                else:
                    if load_shedding_s + load_shedding_e + penalty>20:
                        load_shedding_s = 0
                        load_shedding_e = 0
                        penalty = 20
                    if mark==1:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (
                                    load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 5)
                    else:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 5)
                    reward11t = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 5)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3)

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + (
                                #load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 1e1)
                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (load_shed) * 1e3)
                    #reward1 = -(sum(powerflow2[0]["gen"][:, 1]) +(load_shedding_s+load_shedding_e)*1e3+(load_shed)*1e1)#R1

                    reward2 = (load_shedding_s + load_shedding_e + penalty) * 1 #+ aaa
                    reward2_t = (load_shedding_s + load_shedding_e + penalty) * 1#e2


                #reward2 = -(load_shedding_s + load_shedding_e) * 1e5 - (load_s11 + load_e11) * 10
                reward=reward11
                #reward = reward11 + reward2 - c * 1e5

                record = reward,reward2, reward00, reward11, c, c1,c2, action, attack, load_shedding_s+load_shedding_e,load_shed,powerflow2[0]["success"], action[11],action[12],penalty,aaa,sum(powerflow2[0]["gen"][:, 1]),sum(powerflow[0]["gen"][:, 1])

        return s_, reward, record, 0, reward2,reward2_t,reward11t,reward00t

class IEGS1():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 24  # vlotege 发电机功率，######load shedding
        self.observation_space =43#有功无功负荷

    def reset(self):
        maxPl=np.zeros(shape=(42,25))
        minPl = np.zeros(shape=(42, 25))
        maxPl1 = np.zeros(shape=(42, 25))
        minPl1 = np.ones(shape=(42, 25))
        #rate1= np.array([
        #     1, 1.041666667, 0.912, 0.85, 0.833333333, 0.733333333, 0.75, 1.083333333, 1.466666667, 1.883333333, 1.966666667, 2.216666667, 2.066666667, 1.941666667, 2.116666667,
        #2.233333333, 2.316666667, 2.208333333, 2, 2, 1.833333333, 1.633333333, 1.433333333,1,1
        #])#每小时负荷比例
        rate = np.array([
            0.95, 0.841666667, 0.812, 0.8, 0.7833333333, 0.733333333, 0.75, 0.803333333, 0.866666667, 0.9183333333,
            1.066666667, 1.116666667, 1.166666667, 1.041666667, 1.016666667,
            1.03333333, 1.0516666667, 1.1308333333, 1.166666667, 1.1608333333, 1.16666667, 1.133333333, 1.066666667, 1, 1
        ])  # 每小时负荷比例
        #rate = np.array([
        #    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #    1, 1, 1, 1, 1,
        #    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        #])*rate1[11]  # 每小时负荷比例   13-19(12-18), 18-20(17-19), 17-21(16-20), 16-22(15-21)， 9-23(8-22), 1-24(0-23)
        parameters1 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*1.5
        parameters2 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*0.5
        #负荷节点上下限度
        data=np.array([2,3,4,7,8,10,11,12,14,15,16,17,18,19,20,21,23,24,26,29,30])-1 #负荷节点
        for i in range(25):
            maxPl[:, i] = rate[i] * parameters1
            minPl[:, i] = rate[i] * parameters2
        load_shedding=maxPl-minPl
        state = np.random.uniform(maxPl, minPl)
        state1 = np.random.uniform(maxPl1, minPl1)
        #state1=state1*(maxPl-minPl)
        max_action = np.array(
            [1.05, 1.05, 1.05, 1.05, 1.05, 160, 100, 110, 60, 80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1])
        #max_action = np.array([1.05,1.05,1.05,1.05,1.05,1.05,160,100,110,60,80,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])#前六个是发电机电压控制，7-11是PV发电机输出功率控制，后面1是PQ节点的load shedding
        #min_action = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        min_action = np.array(
            [0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #min_action = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0])
        #max_action=np.array([2,60])
        #min_action=np.array([1,30])
        action_data=max_action-min_action
        action_basic=min_action
        #load_shedding=[]

        #data=GasData()
        return state, action_data, data, action_basic,load_shedding,minPl

    def step(self, action, action_data,data,action_basic,state_,state,step,load_shedding,attack,minPl,mark):
        #state_shedding=np.zeros(shape=(42))
        #state=state*load_shedding[:,step]+minPl[:,step]
        #state_=state_*load_shedding[:,step+1]+minPl[:,step+1]
        PenaltyRate=2
        s_ = state_
        action = (action + 1) / 2
        mark1=np.array(
            [5,7,9,10,11,12,13,14,15,16,17,18,19,20])-1
        action = action * action_data + action_basic





        if attack!=-1:


            attack =attack
            if attack==0:
                attack=7
            elif attack==1:
                attack=8
            elif attack==2:
                attack=9
            elif attack==3:
                attack=12
            elif attack==4:
                attack=16
            elif attack==5:
                attack=17
            elif attack==6:
                attack=18
            elif attack==7:
                attack=19
            elif attack==8:
                attack=20
            elif attack==9:
                attack=21
            elif attack==10:
                attack=22
            elif attack==11:
                attack=23
            elif attack==12:
                attack=24
            elif attack==13:
                attack=25
            elif attack==14:
                attack=28#28
            elif attack==15:
                attack=28
            elif attack==16:
                attack=31
            elif attack==17:
                attack=33
            elif attack==18:
                attack=37
            elif attack==19:
                attack=38
            elif attack==20:
                attack=39

                #if attack[i]==5:
                #    attack[i]=7

                ####################
                #if attack[i]==0:
                #    attack[i]=13
                #if attack[i]==29:
                #    attack[i]=30
                #if attack[i]==15:
                #    attack[i]=14
                #if attack[i]==35:
                #    attack[i]=12
                #if attack[i]==10:
                #    attack[i]=33
                #if attack[i]==40:
                #    attack[i]=12
                #if attack[i]==34:
                #    attack[i]=33
###############################################

            #for N-K attack
            ###################for i in range(len(attack)):
            #    mark=Find(attack[i],attack,i)
            #    if mark!=[]:
            #        p=1
            #        for j in range(len(mark)):
            #            attack[mark[j]]=attack[mark[j]]-p
            #########################            p+=1
        c=0
        c1=0

        load_shedding_s = 0
        penalty = 0
        shedding = np.zeros(shape=(21))


        #if load_shedding!=[]:
        #    for i in range(21):
        #        shedding[i]=load_shedding[i,step]*action[11]
        #    state_shedding[0:21] = state[0:21] - shedding
        #    state_shedding[22:42] = state[22:42]
        #    powerflow2 = runpf(case30(action, state_shedding, data, attack,0,0,0,[]))

        powerflow= runpf(case31(action, state, data,[],0,0,0,[],mark1))

        load_shed=0
        j=0
        for i in mark1:  # for MA and AA
            q = 0
            for p in data:
                if p == data[i]:
                    break
                q += 1
            load_shed += action[j + 10] * state[q]
            j+=1


        if powerflow[0]["success"]==False :
            reward=-1e4
            reward2= np.random.uniform(100, 99.99)
            record = reward, c1, action, attack
            reward2_t=reward2
            reward11t=reward
            reward00t = reward
        else:
            if attack!=-1:
                if attack==12 or attack==33:
                    a=1
                powerflow2 = runpf(case31(action, state, data, attack,[],[],[],[],mark1))

                for i in range(1):
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 1] - powerflow2[0]["gen"][i, 8])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 9] - powerflow2[0]["gen"][i, 1]))) / (
                                       (powerflow2[0]["gen"][i, 8]) - powerflow2[0]["gen"][i, 9])) ** 2
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / (
                                       (powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c1 += ((SolveMax(0,
                                     (powerflow2[0]["gen"][i + 1, 2] - powerflow2[0]["gen"][i + 1, 3])) + SolveMax(
                        0, (
                                powerflow2[0]["gen"][i + 1, 4] - powerflow2[0]["gen"][i + 1, 2]))) / (
                                   (powerflow2[0]["gen"][i + 1, 3]) - powerflow2[0]["gen"][i + 1, 4])) ** 2
                for i in range(30):
                    c1 += ((SolveMax(0, powerflow2[0]["bus"][i, 7] - 1.05) + SolveMax(0,
                                                                                      0.95 - powerflow2[0]["bus"][
                                                                                          i, 7])) / 0.1) ** 2
                # c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c1 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c1 = (c1 / 71) ** 0.5
                sou=int(powerflow2[0]["branch"][int(attack), 0])-1
                load_s = powerflow2[0]["bus"][sou, 2]
                end=int(powerflow2[0]["branch"][int(attack), 1])-1
                load_e = powerflow2[0]["bus"][end, 2]
                load_shedding_s = 0
                load_shedding_e = 0
                load_ss1 = 0
                load_ss2 = 0
                load_ee1 = 0
                load_ee2 = 0
                load_ss = 0
                load_ee = 0
                if sou==26 or end==26 or attack == 21 or attack == 24 or attack == 22 or attack== 23 or attack == 18 or attack == 25:
                    if attack == 21 or attack == 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:
                        if attack == 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                    if sou == 26 or end == 26:
                        powerflow2 = runpf(case31(action, state, data, attack, powerflow2[0]["bus"][28, 2]/2, powerflow2[0]["bus"][29, 2]/2, 28, 29,mark1))
                        load_shedding_s = powerflow2[0]["bus"][28, 2]/2
                        load_shedding_e = powerflow2[0]["bus"][29, 2]/2
                else:
                    for i in range(41):
                        mark111=Find(sou+1,powerflow2[0]["branch"][:, 0],int(attack))#以sou为入口
                        mark2=Find(end+1,powerflow2[0]["branch"][:, 1],int(attack))#以end为出口
                        mark11 = Find(sou+1, powerflow2[0]["branch"][:, 1], int(attack))#以sou为出口
                        mark22 = Find(end+1, powerflow2[0]["branch"][:, 0], int(attack))#以end为入口
                    if mark111!=[]:
                        for i in range(len(mark111)):
                            if powerflow2[0]["branch"][mark111[i], 5]<-powerflow2[0]["branch"][mark111[i], 13]:
                                load_ss1= 0.9*(sum(powerflow2[0]["branch"][mark111[j], 5] for j in range(len(mark111))))
                            else:
                                load_ss1 =0
                    if mark22 != []:
                        for i in range(len(mark22)):
                            if powerflow2[0]["branch"][mark22[i], 5]<-powerflow2[0]["branch"][mark22[i], 15]:
                                load_ee1= 0.9*(sum(powerflow2[0]["branch"][mark22[j], 5] for j in range(len(mark22))))
                            else:
                                load_ee1 =0
                    if mark11 != []:
                        for i in range(len(mark11)):
                            if powerflow2[0]["branch"][mark11[i], 5]<-powerflow2[0]["branch"][mark11[i], 15]:
                                load_ss2= 0.9*(sum(powerflow2[0]["branch"][mark11[j], 5] for j in range(len(mark11))))
                            else:
                                load_ss2 =0
                    if mark2 != []:
                        for i in range(len(mark2)):
                            if powerflow2[0]["branch"][mark2[i], 5]<-powerflow2[0]["branch"][mark2[i], 13]:
                                load_ee2= 0.9*(sum(powerflow2[0]["branch"][mark2[j], 5] for j in range(len(mark2))))
                            else:
                                load_ee2 =0
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss1 =0
                        load_ss2 = 0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee1 = 0
                        load_ee2 = 0
                    if load_ss1!=0 or load_ss2!=0 or load_ee1!=0 or load_ee2!=0:
                        if mark111!=[]:
                            load_ss += load_ss1/ len(mark111)
                        if mark11 != []:
                            load_ss += load_ss2 / len(mark11)
                        if mark22!=[]:
                            load_ee += load_ee1/ len(mark22)
                        if mark2 != []:
                            load_ee += load_ee2 / len(mark2)

                        #load_ss=(load_ss1+load_ss2)/(len(mark1)+len(mark11))
                        #load_ee = (load_ee1 + load_ee2) / (len(mark2) + len(mark22))
                    else:
                        load_ss =load_s
                        load_ee=load_e
                    #if load_ss==0:
                    #    load_ss=0.8*load_s
                    #if load_ee==0:
                    #    load_ee=0.8*load_e
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss =0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee = 0
                    if powerflow2[0]["success"] == True:

                        load_shedding_s = load_s - load_ss
                        load_shedding_e = load_e - load_ee
                        if load_shedding_s < 0:
                            if sou!=1 and sou!=23 and sou!=0 and sou!=22:
                                penalty=-load_shedding_s
                                #penalty=0
                                load_ss= load_s
                                load_shedding_s = 0
                            if sou == 1:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][6, 2] + 0.2 * powerflow2[0]["bus"][7, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.8 * powerflow2[0]["bus"][6, 2], 0.8 * powerflow2[0]["bus"][7, 2], 6, 7,mark1))
                            elif sou == 23:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][21, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.2 * powerflow2[0]["bus"][21, 2], 0, 21, -1,mark1))
                            elif sou == 22:
                                #action1[9] = (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty+=(action[8]-(load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8]))
                                action[8]= (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            elif sou == 0:
                                #action1[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]) / action_data[6] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty +=(action[5]-(load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5]))
                                action[5] = (load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_s = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        elif load_shedding_e < 0:
                            if end != 22:
                                penalty=-load_shedding_e
                                #penalty=0
                                load_ee = load_e
                                load_shedding_e=0
                            if end == 22:
                                #action1[9] = (load_e + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_e = 0
                                #penalty = 0
                                penalty += (action[8] - (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8]))
                                action[8] = (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_e = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        else:
                            if attack==12 or attack==33:
                                a=1
                            else:
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))

                            # if end==26:
                            #    action1[8]=(load_e+powerflow2[0]["branch"][mark2[0], 5]-action_basic)/action_data*20-10
                            #a=1
                            #if load_ss>load_s:
                            #    load_shedding_s=0
                            #if load_ee>load_e:
                            #    load_shedding_e=0

                    if attack == 21 or attack== 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:

                        if attack== 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack== 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack== 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                if attack==12 :
                    load_shedding_s=powerflow[0]["bus"][10, 2]
                    load_shedding_e = 0
                    penalty=0
                if attack==33 :
                    load_shedding_s=1
                    load_shedding_e = powerflow[0]["bus"][25, 2]
                    penalty=0

            for i in range(1):#slack节点有功功率和无功功率是否超出限额
                c+=((SolveMax(0,(powerflow[0]["gen"][i, 1]-powerflow[0]["gen"][i, 8]))+SolveMax(0,(powerflow[0]["gen"][i, 9]-powerflow[0]["gen"][i, 1])))/((powerflow[0]["gen"][i, 8])-powerflow[0]["gen"][i, 9]))**2
                c += ((SolveMax(0, (powerflow[0]["gen"][i, 2] - powerflow[0]["gen"][i, 3])) + SolveMax(0, (powerflow[0]["gen"][i, 4] - powerflow[0]["gen"][i, 2]))) / ((powerflow[0]["gen"][i, 3]) - powerflow[0]["gen"][i, 4])) ** 2
            for i in range(5):#PV节点无功功率是否超出限额
                c += ((SolveMax(0, (powerflow[0]["gen"][i+1, 2] - powerflow[0]["gen"][i+1, 3])) + SolveMax(0, (
                            powerflow[0]["gen"][i+1, 4] - powerflow[0]["gen"][i+1, 2]))) / (
                                  (powerflow[0]["gen"][i+1, 3]) - powerflow[0]["gen"][i+1, 4])) ** 2
            for i in range(30):#节点电压是否超出限额
                c+=((SolveMax(0,powerflow[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow[0]["bus"][i,7]))/0.1)**2
            #c=(c/31)**0.5
            for i in range(41):#线路流量是否超出限额
                c+=(SolveMax(0,powerflow[0]["branch"][i,13]-powerflow[0]["branch"][i,5])/powerflow[0]["branch"][i,5])**2
            c = (c / 72) ** 0.5


            ###
            c2=0

            if attack!=-1:
                for i in range(1):
                    c2+=((SolveMax(0,(powerflow2[0]["gen"][i, 1]-powerflow2[0]["gen"][i, 8]))+SolveMax(0,(powerflow2[0]["gen"][i, 9]-powerflow2[0]["gen"][i, 1])))/((powerflow2[0]["gen"][i, 8])-powerflow2[0]["gen"][i, 9]))**2
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / ((powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i+1, 2] - powerflow2[0]["gen"][i+1, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i+1, 4] - powerflow2[0]["gen"][i+1, 2]))) / (
                                      (powerflow2[0]["gen"][i+1, 3]) - powerflow2[0]["gen"][i+1, 4])) ** 2
                for i in range(30):
                    c2+=((SolveMax(0,powerflow2[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow2[0]["bus"][i,7]))/0.1)**2
                #c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c2 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c2 = (c2 / 71) ** 0.5
            if c1==0:
                load_shedding_s+=c2*1e5

            ###


            if powerflow[0]["success"]==False :
                reward0=-1e8
                reward00 = -1e8
            else:
                reward00 = -(sum(powerflow[0]["gen"][:, 1])+c*5e3)
                reward0 = -(sum(powerflow[0]["gen"][:, 1]) )#R0
            aaa = np.random.uniform(1000, 999.99)
            if attack==-1:
                reward=reward00
                reward2=0
                reward1=[]
                record = reward, reward0, reward1, c, action
                reward00t=0
                reward2_t=0
                reward11t=0
            else:
                if powerflow2[0]["success"]==False :
                    if mark==1:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3 + (load_shed) * 5 + 10000)
                    else:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3+ (load_shed) * 5 + 2000)
                    reward11t = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3 + (load_shed) * 5 + 2000)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3 )


                    #if mark==1:
                    #    reward11 = -(sum(powerflow[0]["gen"][:, 1])  + (load_shed) * 5 + 10000)
                    #else:
                    #    reward11 = -(sum(powerflow[0]["gen"][:, 1]) + (load_shed) * 5 + 2000)
                    #reward11t = -(sum(powerflow[0]["gen"][:, 1]) + (load_shed) * 5 + 2000)
                    #reward00t = -(sum(powerflow[0]["gen"][:, 1]) )



                    reward2 =10
                    reward2_t =20
                else:
                    #if load_shedding_s + load_shedding_e + penalty>20:
                    #    load_shedding_s = 0
                    #    load_shedding_e = 0
                    #    penalty = 20


                    if mark==1:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 5*PenaltyRate + (load_shed) * 5)
                    else:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 5*PenaltyRate + (load_shed) * 5)
                    reward11t = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                load_shedding_s + load_shedding_e + penalty) * 5*PenaltyRate + (load_shed) * 5)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3)

                    #if mark == 1:
                    #    reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + (
                    #            load_shedding_s + load_shedding_e + penalty) * 5 * PenaltyRate + (load_shed) * 5)
                    #else:
                    #    reward11 = -(sum(powerflow2[0]["gen"][:, 1])  + (
                    #            load_shedding_s + load_shedding_e + penalty) * 5 * PenaltyRate + (load_shed) * 5)
                    #reward11t = -(sum(powerflow2[0]["gen"][:, 1]) +  (
                    #        load_shedding_s + load_shedding_e + penalty) * 5 * PenaltyRate + (load_shed) * 5)
                    #reward00t = -(sum(powerflow[0]["gen"][:, 1])  )

                    reward2 = (load_shedding_s + load_shedding_e + penalty) * 1 #+ aaa
                    reward2_t = (load_shedding_s + load_shedding_e + penalty) * 1#e2


                #reward2 = -(load_shedding_s + load_shedding_e) * 1e5 - (load_s11 + load_e11) * 10
                reward=reward11
                #reward = reward11 + reward2 - c * 1e5
                #record = reward, reward2,  c2,  attack, load_shedding_s + load_shedding_e, load_shed, powerflow2[0]["success"], penalty, sum(powerflow[0]["gen"][:, 1])

                record = reward,reward2, reward00, reward11, c, c1,c2, action, attack, load_shedding_s+load_shedding_e,load_shed,powerflow2[0]["success"],penalty,sum(powerflow2[0]["gen"][:, 1]),sum(powerflow[0]["gen"][:, 1])

        return s_, reward, record, 0, reward2,reward2_t,reward11t,reward00t

class IEGS2():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 24  # vlotege 发电机功率，######load shedding
        self.observation_space =42#有功无功负荷

    def reset(self):
        maxPl=np.zeros(shape=(42,25))
        minPl = np.zeros(shape=(42, 25))
        maxPl1 = np.zeros(shape=(42, 25))
        minPl1 = np.ones(shape=(42, 25))
        rate= np.array([
             1, 1.041666667, 0.912, 0.85, 0.833333333, 0.733333333, 0.75, 1.083333333, 1.466666667, 1.883333333, 1.966666667, 2.216666667, 2.066666667, 1.941666667, 2.116666667,
        2.233333333, 2.316666667, 2.208333333, 2, 2, 1.833333333, 1.633333333, 1.433333333,1,1
        ])#每小时负荷比例
        parameters1 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*1.5
        parameters2 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*0.5
        #负荷节点上下限度
        data=np.array([2,3,4,7,8,10,11,12,14,15,16,17,18,19,20,21,23,24,26,29,30])-1 #负荷节点
        for i in range(25):
            maxPl[:, i] = rate[i] * parameters1
            minPl[:, i] = rate[i] * parameters2
        load_shedding=maxPl-minPl
        state = np.random.uniform(maxPl, minPl)
        state1 = np.random.uniform(maxPl1, minPl1)
        #state1=state1*(maxPl-minPl)
        max_action = np.array(
            [1.05, 1.05, 1.05, 1.05, 1.05, 160, 100, 110, 60, 80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1])
        #max_action = np.array([1.05,1.05,1.05,1.05,1.05,1.05,160,100,110,60,80,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])#前六个是发电机电压控制，7-11是PV发电机输出功率控制，后面1是PQ节点的load shedding
        #min_action = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        min_action = np.array(
            [0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #min_action = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0])
        #max_action=np.array([2,60])
        #min_action=np.array([1,30])
        action_data=max_action-min_action
        action_basic=min_action
        #load_shedding=[]

        #data=GasData()
        return state, action_data, data, action_basic,load_shedding,minPl

    def step(self, action, action_data,data,action_basic,state_,state,step,load_shedding,attack,minPl,mark):
        #state_shedding=np.zeros(shape=(42))
        #state=state*load_shedding[:,step]+minPl[:,step]
        #state_=state_*load_shedding[:,step+1]+minPl[:,step+1]
        s_ = state_
        action = (action + 10) / 20
        mark1=np.array(
            [5,7,9,10,11,12,13,14,15,16,17,18,19,20])-1
             #10 12 15 16 17 18 19 20 21 23 24 26 29 30
        action = action * action_data + action_basic





        if attack!=-1:


            attack =attack
            if attack==0:
                attack=7
            elif attack==1:
                attack=8
            elif attack==2:
                attack=9
            elif attack==3:
                attack=12
            elif attack==4:
                attack=16
            elif attack==5:
                attack=17
            elif attack==6:
                attack=18
            elif attack==7:
                attack=19
            elif attack==8:
                attack=20
            elif attack==9:
                attack=21
            elif attack==10:
                attack=22
            elif attack==11:
                attack=23
            elif attack==12:
                attack=24
            elif attack==13:
                attack=25
            elif attack==14:
                attack=28#28
            elif attack==15:
                attack=28
            elif attack==16:
                attack=31
            elif attack==17:
                attack=33
            elif attack==18:
                attack=37
            elif attack==19:
                attack=38
            elif attack==20:
                attack=39

                #if attack[i]==5:
                #    attack[i]=7

                ####################
                #if attack[i]==0:
                #    attack[i]=13
                #if attack[i]==29:
                #    attack[i]=30
                #if attack[i]==15:
                #    attack[i]=14
                #if attack[i]==35:
                #    attack[i]=12
                #if attack[i]==10:
                #    attack[i]=33
                #if attack[i]==40:
                #    attack[i]=12
                #if attack[i]==34:
                #    attack[i]=33
###############################################

            #for N-K attack
            ###################for i in range(len(attack)):
            #    mark=Find(attack[i],attack,i)
            #    if mark!=[]:
            #        p=1
            #        for j in range(len(mark)):
            #            attack[mark[j]]=attack[mark[j]]-p
            #########################            p+=1
        c=0
        c1=0

        load_shedding_s = 0
        penalty = 0
        shedding = np.zeros(shape=(21))


        #if load_shedding!=[]:
        #    for i in range(21):
        #        shedding[i]=load_shedding[i,step]*action[11]
        #    state_shedding[0:21] = state[0:21] - shedding
        #    state_shedding[22:42] = state[22:42]
        #    powerflow2 = runpf(case30(action, state_shedding, data, attack,0,0,0,[]))

        powerflow= runpf(case31(action, state, data,[],0,0,0,[],mark1))

        load_shed=0
        j=0
        for i in mark1:  # for MA and AA
            q = 0
            for p in data:
                if p == data[i]:
                    break
                q += 1
            load_shed += action[j + 10] * state[q]
            j+=1


        if powerflow[0]["success"]==False :
            reward=-1e4
            reward2= np.random.uniform(100, 99.99)
            record = reward, c1, action, attack
            reward2_t=reward2
            reward11t=reward
            reward00t = reward
        else:
            if attack!=-1:
                if attack==12 or attack==33:
                    a=1
                powerflow2 = runpf(case31(action, state, data, attack,[],[],[],[],mark1))

                for i in range(1):
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 1] - powerflow2[0]["gen"][i, 8])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 9] - powerflow2[0]["gen"][i, 1]))) / (
                                       (powerflow2[0]["gen"][i, 8]) - powerflow2[0]["gen"][i, 9])) ** 2
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / (
                                       (powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c1 += ((SolveMax(0,
                                     (powerflow2[0]["gen"][i + 1, 2] - powerflow2[0]["gen"][i + 1, 3])) + SolveMax(
                        0, (
                                powerflow2[0]["gen"][i + 1, 4] - powerflow2[0]["gen"][i + 1, 2]))) / (
                                   (powerflow2[0]["gen"][i + 1, 3]) - powerflow2[0]["gen"][i + 1, 4])) ** 2
                for i in range(30):
                    c1 += ((SolveMax(0, powerflow2[0]["bus"][i, 7] - 1.05) + SolveMax(0,
                                                                                      0.95 - powerflow2[0]["bus"][
                                                                                          i, 7])) / 0.1) ** 2
                # c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c1 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c1 = (c1 / 71) ** 0.5
                sou=int(powerflow2[0]["branch"][int(attack), 0])-1
                load_s = powerflow2[0]["bus"][sou, 2]
                end=int(powerflow2[0]["branch"][int(attack), 1])-1
                load_e = powerflow2[0]["bus"][end, 2]
                load_shedding_s = 0
                load_shedding_e = 0
                load_ss1 = 0
                load_ss2 = 0
                load_ee1 = 0
                load_ee2 = 0
                load_ss = 0
                load_ee = 0
                if sou==26 or end==26 or attack == 21 or attack == 24 or attack == 22 or attack== 23 or attack == 18 or attack == 25:
                    if attack == 21 or attack == 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:
                        if attack == 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                    if sou == 26 or end == 26:
                        powerflow2 = runpf(case31(action, state, data, attack, powerflow2[0]["bus"][28, 2]/2, powerflow2[0]["bus"][29, 2]/2, 28, 29,mark1))
                        load_shedding_s = powerflow2[0]["bus"][28, 2]/2
                        load_shedding_e = powerflow2[0]["bus"][29, 2]/2
                else:
                    for i in range(41):
                        mark111=Find(sou+1,powerflow2[0]["branch"][:, 0],int(attack))#以sou为入口
                        mark2=Find(end+1,powerflow2[0]["branch"][:, 1],int(attack))#以end为出口
                        mark11 = Find(sou+1, powerflow2[0]["branch"][:, 1], int(attack))#以sou为出口
                        mark22 = Find(end+1, powerflow2[0]["branch"][:, 0], int(attack))#以end为入口
                    if mark111!=[]:
                        for i in range(len(mark111)):
                            if powerflow2[0]["branch"][mark111[i], 5]<-powerflow2[0]["branch"][mark111[i], 13]:
                                load_ss1= 0.9*(sum(powerflow2[0]["branch"][mark111[j], 5] for j in range(len(mark111))))
                            else:
                                load_ss1 =0
                    if mark22 != []:
                        for i in range(len(mark22)):
                            if powerflow2[0]["branch"][mark22[i], 5]<-powerflow2[0]["branch"][mark22[i], 15]:
                                load_ee1= 0.9*(sum(powerflow2[0]["branch"][mark22[j], 5] for j in range(len(mark22))))
                            else:
                                load_ee1 =0
                    if mark11 != []:
                        for i in range(len(mark11)):
                            if powerflow2[0]["branch"][mark11[i], 5]<-powerflow2[0]["branch"][mark11[i], 15]:
                                load_ss2= 0.9*(sum(powerflow2[0]["branch"][mark11[j], 5] for j in range(len(mark11))))
                            else:
                                load_ss2 =0
                    if mark2 != []:
                        for i in range(len(mark2)):
                            if powerflow2[0]["branch"][mark2[i], 5]<-powerflow2[0]["branch"][mark2[i], 13]:
                                load_ee2= 0.9*(sum(powerflow2[0]["branch"][mark2[j], 5] for j in range(len(mark2))))
                            else:
                                load_ee2 =0
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss1 =0
                        load_ss2 = 0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee1 = 0
                        load_ee2 = 0
                    if load_ss1!=0 or load_ss2!=0 or load_ee1!=0 or load_ee2!=0:
                        if mark111!=[]:
                            load_ss += load_ss1/ len(mark111)
                        if mark11 != []:
                            load_ss += load_ss2 / len(mark11)
                        if mark22!=[]:
                            load_ee += load_ee1/ len(mark22)
                        if mark2 != []:
                            load_ee += load_ee2 / len(mark2)

                        #load_ss=(load_ss1+load_ss2)/(len(mark1)+len(mark11))
                        #load_ee = (load_ee1 + load_ee2) / (len(mark2) + len(mark22))
                    else:
                        load_ss =load_s
                        load_ee=load_e
                    #if load_ss==0:
                    #    load_ss=0.8*load_s
                    #if load_ee==0:
                    #    load_ee=0.8*load_e
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss =0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee = 0
                    if powerflow2[0]["success"] == True:

                        load_shedding_s = load_s - load_ss
                        load_shedding_e = load_e - load_ee
                        if load_shedding_s < 0:
                            if sou!=1 and sou!=23 and sou!=0 and sou!=22:
                                penalty=-load_shedding_s
                                #penalty=0
                                load_ss= load_s
                                load_shedding_s = 0
                            if sou == 1:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][6, 2] + 0.2 * powerflow2[0]["bus"][7, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.8 * powerflow2[0]["bus"][6, 2], 0.8 * powerflow2[0]["bus"][7, 2], 6, 7,mark1))
                            elif sou == 23:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][21, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.2 * powerflow2[0]["bus"][21, 2], 0, 21, -1,mark1))
                            elif sou == 22:
                                #action1[9] = (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty+=(action[8]-(load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8]))
                                action[8]= (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            elif sou == 0:
                                #action1[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]) / action_data[6] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty +=(action[5]-(load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5]))
                                action[5] = (load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_s = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        elif load_shedding_e < 0:
                            if end != 22:
                                penalty=-load_shedding_e
                                #penalty=0
                                load_ee = load_e
                                load_shedding_e=0
                            if end == 22:
                                #action1[9] = (load_e + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_e = 0
                                #penalty = 0
                                penalty += (action[8] - (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8]))
                                action[8] = (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_e = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        else:
                            if attack==12 or attack==33:
                                a=1
                            else:
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))

                            # if end==26:
                            #    action1[8]=(load_e+powerflow2[0]["branch"][mark2[0], 5]-action_basic)/action_data*20-10
                            #a=1
                            #if load_ss>load_s:
                            #    load_shedding_s=0
                            #if load_ee>load_e:
                            #    load_shedding_e=0

                    if attack == 21 or attack== 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:

                        if attack== 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack== 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack== 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                if attack==12 :
                    load_shedding_s=powerflow[0]["bus"][10, 2]
                    load_shedding_e = 0
                    penalty=0
                if attack==33 :
                    load_shedding_s=1
                    load_shedding_e = powerflow[0]["bus"][25, 2]
                    penalty=0
            for i in range(1):#slack节点有功功率和无功功率是否超出限额
                c+=((SolveMax(0,(powerflow[0]["gen"][i, 1]-powerflow[0]["gen"][i, 8]))+SolveMax(0,(powerflow[0]["gen"][i, 9]-powerflow[0]["gen"][i, 1])))/((powerflow[0]["gen"][i, 8])-powerflow[0]["gen"][i, 9]))**2
                c += ((SolveMax(0, (powerflow[0]["gen"][i, 2] - powerflow[0]["gen"][i, 3])) + SolveMax(0, (powerflow[0]["gen"][i, 4] - powerflow[0]["gen"][i, 2]))) / ((powerflow[0]["gen"][i, 3]) - powerflow[0]["gen"][i, 4])) ** 2
            for i in range(5):#PV节点无功功率是否超出限额
                c += ((SolveMax(0, (powerflow[0]["gen"][i+1, 2] - powerflow[0]["gen"][i+1, 3])) + SolveMax(0, (
                            powerflow[0]["gen"][i+1, 4] - powerflow[0]["gen"][i+1, 2]))) / (
                                  (powerflow[0]["gen"][i+1, 3]) - powerflow[0]["gen"][i+1, 4])) ** 2
            for i in range(30):#节点电压是否超出限额
                c+=((SolveMax(0,powerflow[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow[0]["bus"][i,7]))/0.1)**2
            #c=(c/31)**0.5
            for i in range(41):#线路流量是否超出限额
                c+=(SolveMax(0,powerflow[0]["branch"][i,13]-powerflow[0]["branch"][i,5])/powerflow[0]["branch"][i,5])**2
            c = (c / 72) ** 0.5


            ###
            c2=0

            if attack!=-1:
                for i in range(1):
                    c2+=((SolveMax(0,(powerflow2[0]["gen"][i, 1]-powerflow2[0]["gen"][i, 8]))+SolveMax(0,(powerflow2[0]["gen"][i, 9]-powerflow2[0]["gen"][i, 1])))/((powerflow2[0]["gen"][i, 8])-powerflow2[0]["gen"][i, 9]))**2
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / ((powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i+1, 2] - powerflow2[0]["gen"][i+1, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i+1, 4] - powerflow2[0]["gen"][i+1, 2]))) / (
                                      (powerflow2[0]["gen"][i+1, 3]) - powerflow2[0]["gen"][i+1, 4])) ** 2
                for i in range(30):
                    c2+=((SolveMax(0,powerflow2[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow2[0]["bus"][i,7]))/0.1)**2
                #c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c2 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c2 = (c2 / 71) ** 0.5
            if c1==0:
                load_shedding_s+=c2*1e5

            ###


            if powerflow[0]["success"]==False :
                reward0=-1e8
                reward00 = -1e8
            else:
                reward00 = -(sum(powerflow[0]["gen"][:, 1])+c*1e3)
                reward0 = -(sum(powerflow[0]["gen"][:, 1]) )#R0
            aaa = np.random.uniform(1000, 999.99)
            if attack==-1:
                reward=reward00
                reward2=0
                reward1=[]
                record = reward, reward0, reward1, c, action
                reward00t=0
                reward2_t=0
                reward11t=0
            else:
                if powerflow2[0]["success"]==False :
                    if mark==1:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e4 + (load_shed) * 1 + 10000)
                    else:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 1 + 2000)
                    reward11t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 1 + 2000)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 )
                    #reward1=-1e6
                    #reward11 = -1e6

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1])  + (load_shed) * 1e2)
                    reward2 =20
                    reward2_t =20
                else:
                    if load_shedding_s + load_shedding_e + penalty>20:
                        load_shedding_s = 0
                        load_shedding_e = 0
                        penalty = 20
                    if mark==1:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (
                                    load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 1)
                    else:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 1)
                    reward11t = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 1)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3)

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + (
                                #load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 1e1)
                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (load_shed) * 1e3)
                    #reward1 = -(sum(powerflow2[0]["gen"][:, 1]) +(load_shedding_s+load_shedding_e)*1e3+(load_shed)*1e1)#R1

                    reward2 = (load_shedding_s + load_shedding_e + penalty) * 1 #+ aaa
                    reward2_t = (load_shedding_s + load_shedding_e + penalty) * 1#e2


                #reward2 = -(load_shedding_s + load_shedding_e) * 1e5 - (load_s11 + load_e11) * 10
                reward=reward11
                #reward = reward11 + reward2 - c * 1e5

                record = reward,reward2, reward00, reward11, c, c1,c2, action, attack, load_shedding_s+load_shedding_e,load_shed,powerflow2[0]["success"], action[11],action[12],penalty,aaa,sum(powerflow2[0]["gen"][:, 1]),sum(powerflow[0]["gen"][:, 1])

        return s_, reward, record, 0, reward2,reward2_t,reward11t,reward00t

class IEGS3():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 24  # vlotege 发电机功率，######load shedding
        self.observation_space =42#有功无功负荷

    def reset(self):
        maxPl=np.zeros(shape=(42,25))
        minPl = np.zeros(shape=(42, 25))
        maxPl1 = np.zeros(shape=(42, 25))
        minPl1 = np.ones(shape=(42, 25))
        rate= np.array([
            0.95, 0.841666667, 0.812, 0.8, 0.7833333333, 0.733333333, 0.75, 0.803333333, 0.866666667, 0.9183333333,
            1.066666667, 1.116666667, 1.166666667, 1.041666667, 1.016666667,
            1.03333333, 1.0516666667, 1.1308333333, 1.166666667, 1.1608333333, 1.16666667, 1.133333333, 1.066666667, 1, 1
        ])#每小时负荷比例
        parameters1 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*1.5
        parameters2 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*0.5
        #负荷节点上下限度
        data=np.array([2,3,4,7,8,10,11,12,14,15,16,17,18,19,20,21,23,24,26,29,30])-1 #负荷节点
        for i in range(25):
            maxPl[:, i] = rate[i] * parameters1
            minPl[:, i] = rate[i] * parameters2
        load_shedding=maxPl-minPl
        state = np.random.uniform(maxPl, minPl)
        state1 = np.random.uniform(maxPl1, minPl1)
        #state1=state1*(maxPl-minPl)
        max_action = np.array(
            [1.05, 1.05, 1.05, 1.05, 1.05, 160, 100, 110, 60, 80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1])
        #max_action = np.array([1.05,1.05,1.05,1.05,1.05,1.05,160,100,110,60,80,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])#前六个是发电机电压控制，7-11是PV发电机输出功率控制，后面1是PQ节点的load shedding
        #min_action = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        min_action = np.array(
            [0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #min_action = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0])
        #max_action=np.array([2,60])
        #min_action=np.array([1,30])
        action_data=max_action-min_action
        action_basic=min_action
        #load_shedding=[]

        #data=GasData()
        return state, action_data, data, action_basic,load_shedding,minPl

    def step(self, action, action_data,data,action_basic,state_,state,step,load_shedding,attack,minPl,mark):
        penalty1 = 0
        #state_shedding=np.zeros(shape=(42))
        #state=state*load_shedding[:,step]+minPl[:,step]
        #state_=state_*load_shedding[:,step+1]+minPl[:,step+1]
        s_ = state_
        action = (action + 10) / 20
        mark1=np.array(
            [5,7,9,10,11,12,13,14,15,16,17,18,19,20])-1
        action = action * action_data + action_basic





        if attack!=-1:


            attack =attack
            if attack==0:
                attack=7
            elif attack==1:
                attack=8
            elif attack==2:
                attack=9
            elif attack==3:
                attack=12
            elif attack==4:
                attack=16
            elif attack==5:
                attack=17
            elif attack==6:
                attack=18
            elif attack==7:
                attack=19
            elif attack==8:
                attack=20
            elif attack==9:
                attack=21
            elif attack==10:
                attack=22
            elif attack==11:
                attack=23
            elif attack==12:
                attack=24
            elif attack==13:
                attack=25
            elif attack==14:
                attack=28#28
            elif attack==15:
                attack=28
            elif attack==16:
                attack=31
            elif attack==17:
                attack=33
            elif attack==18:
                attack=37
            elif attack==19:
                attack=38
            elif attack==20:
                attack=39

                #if attack[i]==5:
                #    attack[i]=7

                ####################
                #if attack[i]==0:
                #    attack[i]=13
                #if attack[i]==29:
                #    attack[i]=30
                #if attack[i]==15:
                #    attack[i]=14
                #if attack[i]==35:
                #    attack[i]=12
                #if attack[i]==10:
                #    attack[i]=33
                #if attack[i]==40:
                #    attack[i]=12
                #if attack[i]==34:
                #    attack[i]=33
###############################################

            #for N-K attack
            ###################for i in range(len(attack)):
            #    mark=Find(attack[i],attack,i)
            #    if mark!=[]:
            #        p=1
            #        for j in range(len(mark)):
            #            attack[mark[j]]=attack[mark[j]]-p
            #########################            p+=1
        c=0
        c1=0

        load_shedding_s = 0
        penalty = 0
        shedding = np.zeros(shape=(21))


        #if load_shedding!=[]:
        #    for i in range(21):
        #        shedding[i]=load_shedding[i,step]*action[11]
        #    state_shedding[0:21] = state[0:21] - shedding
        #    state_shedding[22:42] = state[22:42]
        #    powerflow2 = runpf(case30(action, state_shedding, data, attack,0,0,0,[]))

        powerflow= runpf(case31(action, state, data,[],0,0,0,[],mark1))

        load_shed=0
        j=0
        for i in mark1:  # for MA and AA
            q = 0
            for p in data:
                if p == data[i]:
                    break
                q += 1
            load_shed += action[j + 10] * state[q]
            j+=1


        if powerflow[0]["success"]==False :
            reward=-1e4
            reward2= np.random.uniform(100, 99.99)
            record = reward, c1, action, attack
            reward2_t=reward2
            reward11t=reward
            reward00t = reward
        else:
            if attack!=-1:
                if attack==12 or attack==33:
                    a=1
                powerflow2 = runpf(case31(action, state, data, attack,[],[],[],[],mark1))

                for i in range(1):
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 1] - powerflow2[0]["gen"][i, 8])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 9] - powerflow2[0]["gen"][i, 1]))) / (
                                       (powerflow2[0]["gen"][i, 8]) - powerflow2[0]["gen"][i, 9])) ** 2
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / (
                                       (powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c1 += ((SolveMax(0,
                                     (powerflow2[0]["gen"][i + 1, 2] - powerflow2[0]["gen"][i + 1, 3])) + SolveMax(
                        0, (
                                powerflow2[0]["gen"][i + 1, 4] - powerflow2[0]["gen"][i + 1, 2]))) / (
                                   (powerflow2[0]["gen"][i + 1, 3]) - powerflow2[0]["gen"][i + 1, 4])) ** 2
                for i in range(30):
                    c1 += ((SolveMax(0, powerflow2[0]["bus"][i, 7] - 1.05) + SolveMax(0,
                                                                                      0.95 - powerflow2[0]["bus"][
                                                                                          i, 7])) / 0.1) ** 2
                # c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c1 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c1 = (c1 / 71) ** 0.5
                sou=int(powerflow2[0]["branch"][int(attack), 0])-1
                load_s = powerflow2[0]["bus"][sou, 2]
                end=int(powerflow2[0]["branch"][int(attack), 1])-1
                load_e = powerflow2[0]["bus"][end, 2]
                load_shedding_s = 0
                load_shedding_e = 0
                load_ss1 = 0
                load_ss2 = 0
                load_ee1 = 0
                load_ee2 = 0
                load_ss = 0
                load_ee = 0
                if sou==26 or end==26 or attack == 21 or attack == 24 or attack == 22 or attack== 23 or attack == 18 or attack == 25:
                    if attack == 21 or attack == 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:
                        if attack == 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                    if sou == 26 or end == 26:
                        powerflow2 = runpf(case31(action, state, data, attack, powerflow2[0]["bus"][28, 2]/2, powerflow2[0]["bus"][29, 2]/2, 28, 29,mark1))
                        load_shedding_s = powerflow2[0]["bus"][28, 2]/2
                        load_shedding_e = powerflow2[0]["bus"][29, 2]/2
                else:
                    for i in range(41):
                        mark111=Find(sou+1,powerflow2[0]["branch"][:, 0],int(attack))#以sou为入口
                        mark2=Find(end+1,powerflow2[0]["branch"][:, 1],int(attack))#以end为出口
                        mark11 = Find(sou+1, powerflow2[0]["branch"][:, 1], int(attack))#以sou为出口
                        mark22 = Find(end+1, powerflow2[0]["branch"][:, 0], int(attack))#以end为入口
                    if mark111!=[]:
                        for i in range(len(mark111)):
                            if powerflow2[0]["branch"][mark111[i], 5]<-powerflow2[0]["branch"][mark111[i], 13]:
                                load_ss1= 0.9*(sum(powerflow2[0]["branch"][mark111[j], 5] for j in range(len(mark111))))
                            else:
                                load_ss1 =0
                    if mark22 != []:
                        for i in range(len(mark22)):
                            if powerflow2[0]["branch"][mark22[i], 5]<-powerflow2[0]["branch"][mark22[i], 15]:
                                load_ee1= 0.9*(sum(powerflow2[0]["branch"][mark22[j], 5] for j in range(len(mark22))))
                            else:
                                load_ee1 =0
                    if mark11 != []:
                        for i in range(len(mark11)):
                            if powerflow2[0]["branch"][mark11[i], 5]<-powerflow2[0]["branch"][mark11[i], 15]:
                                load_ss2= 0.9*(sum(powerflow2[0]["branch"][mark11[j], 5] for j in range(len(mark11))))
                            else:
                                load_ss2 =0
                    if mark2 != []:
                        for i in range(len(mark2)):
                            if powerflow2[0]["branch"][mark2[i], 5]<-powerflow2[0]["branch"][mark2[i], 13]:
                                load_ee2= 0.9*(sum(powerflow2[0]["branch"][mark2[j], 5] for j in range(len(mark2))))
                            else:
                                load_ee2 =0
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss1 =0
                        load_ss2 = 0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee1 = 0
                        load_ee2 = 0
                    if load_ss1!=0 or load_ss2!=0 or load_ee1!=0 or load_ee2!=0:
                        if mark111!=[]:
                            load_ss += load_ss1/ len(mark111)
                        if mark11 != []:
                            load_ss += load_ss2 / len(mark11)
                        if mark22!=[]:
                            load_ee += load_ee1/ len(mark22)
                        if mark2 != []:
                            load_ee += load_ee2 / len(mark2)

                        #load_ss=(load_ss1+load_ss2)/(len(mark1)+len(mark11))
                        #load_ee = (load_ee1 + load_ee2) / (len(mark2) + len(mark22))
                    else:
                        load_ss =load_s
                        load_ee=load_e
                    #if load_ss==0:
                    #    load_ss=0.8*load_s
                    #if load_ee==0:
                    #    load_ee=0.8*load_e
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss =0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee = 0
                    if powerflow2[0]["success"] == True:

                        load_shedding_s = load_s - load_ss
                        load_shedding_e = load_e - load_ee
                        if load_shedding_s < 0:
                            if sou!=1 and sou!=23 and sou!=0 and sou!=22:
                                penalty=-load_shedding_s
                                #penalty=0
                                load_ss= load_s
                                load_shedding_s = 0
                            if sou == 1:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][6, 2] + 0.2 * powerflow2[0]["bus"][7, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.8 * powerflow2[0]["bus"][6, 2], 0.8 * powerflow2[0]["bus"][7, 2], 6, 7,mark1))
                            elif sou == 23:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][21, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.2 * powerflow2[0]["bus"][21, 2], 0, 21, -1,mark1))
                            elif sou == 22:
                                #action1[9] = (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty+=(action[8]-(load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8]))
                                action[8]= (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            elif sou == 0:
                                #action1[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]) / action_data[6] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty +=(action[5]-(load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5]))
                                action[5] = (load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_s = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        elif load_shedding_e < 0:
                            if end != 22:
                                penalty=-load_shedding_e
                                #penalty=0
                                load_ee = load_e
                                load_shedding_e=0
                            if end == 22:
                                #action1[9] = (load_e + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_e = 0
                                #penalty = 0
                                penalty += (action[8] - (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8]))
                                action[8] = (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_e = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        else:
                            if attack==12 or attack==33:
                                a=1
                            else:
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))

                            # if end==26:
                            #    action1[8]=(load_e+powerflow2[0]["branch"][mark2[0], 5]-action_basic)/action_data*20-10
                            #a=1
                            #if load_ss>load_s:
                            #    load_shedding_s=0
                            #if load_ee>load_e:
                            #    load_shedding_e=0

                    if attack == 21 or attack== 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:

                        if attack== 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack== 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack== 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                if attack==12 :
                    load_shedding_s=powerflow[0]["bus"][10, 2]
                    load_shedding_e = 0
                    penalty=0
                if attack==33 :
                    load_shedding_s=1
                    load_shedding_e = powerflow[0]["bus"][25, 2]
                    penalty=0
            for i in range(1):#slack节点有功功率和无功功率是否超出限额
                c+=((SolveMax(0,(powerflow[0]["gen"][i, 1]-powerflow[0]["gen"][i, 8]))+SolveMax(0,(powerflow[0]["gen"][i, 9]-powerflow[0]["gen"][i, 1])))/((powerflow[0]["gen"][i, 8])-powerflow[0]["gen"][i, 9]))**2
                c += ((SolveMax(0, (powerflow[0]["gen"][i, 2] - powerflow[0]["gen"][i, 3])) + SolveMax(0, (powerflow[0]["gen"][i, 4] - powerflow[0]["gen"][i, 2]))) / ((powerflow[0]["gen"][i, 3]) - powerflow[0]["gen"][i, 4])) ** 2
            for i in range(5):#PV节点无功功率是否超出限额
                c += ((SolveMax(0, (powerflow[0]["gen"][i+1, 2] - powerflow[0]["gen"][i+1, 3])) + SolveMax(0, (
                            powerflow[0]["gen"][i+1, 4] - powerflow[0]["gen"][i+1, 2]))) / (
                                  (powerflow[0]["gen"][i+1, 3]) - powerflow[0]["gen"][i+1, 4])) ** 2
            for i in range(30):#节点电压是否超出限额
                c+=((SolveMax(0,powerflow[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow[0]["bus"][i,7]))/0.1)**2
            #c=(c/31)**0.5
            for i in range(41):#线路流量是否超出限额
                c+=(SolveMax(0,powerflow[0]["branch"][i,13]-powerflow[0]["branch"][i,5])/powerflow[0]["branch"][i,5])**2
            c = (c / 72) ** 0.5


            ###
            c2=0

            if attack!=-1:
                for i in range(1):
                    c2+=((SolveMax(0,(powerflow2[0]["gen"][i, 1]-powerflow2[0]["gen"][i, 8]))+SolveMax(0,(powerflow2[0]["gen"][i, 9]-powerflow2[0]["gen"][i, 1])))/((powerflow2[0]["gen"][i, 8])-powerflow2[0]["gen"][i, 9]))**2
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / ((powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i+1, 2] - powerflow2[0]["gen"][i+1, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i+1, 4] - powerflow2[0]["gen"][i+1, 2]))) / (
                                      (powerflow2[0]["gen"][i+1, 3]) - powerflow2[0]["gen"][i+1, 4])) ** 2
                for i in range(30):
                    c2+=((SolveMax(0,powerflow2[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow2[0]["bus"][i,7]))/0.1)**2
                #c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c2 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c2 = (c2 / 71) ** 0.5
            if c1==0:
                load_shedding_s+=c2*1e5

            ###


            if powerflow[0]["success"]==False :
                reward0=-1e8
                reward00 = -1e8
            else:
                reward00 = -(sum(powerflow[0]["gen"][:, 1])+c*1e3)
                reward0 = -(sum(powerflow[0]["gen"][:, 1]) )#R0
            aaa = np.random.uniform(1000, 999.99)
            if attack==-1:
                reward=reward00
                reward2=0
                reward1=[]
                record = reward, reward0, reward1, c, action
                reward00t=0
                reward2_t=0
                reward11t=0
            else:
                if powerflow2[0]["success"]==False :
                    if mark==1:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 5 + 10000)
                    else:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3+ (load_shed) * 5 + 3000)
                    reward11t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 5 + 3000)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 )
                    #reward1=-1e6
                    #reward11 = -1e6

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1])  + (load_shed) * 1e2)
                    reward2 =10
                    reward2_t =20
                else:
                    if load_shedding_s + load_shedding_e + penalty>20:
                        penalty1=load_shedding_s + load_shedding_e + penalty
                        load_shedding_s = 0
                        load_shedding_e = 0
                        penalty = 20
                    if mark==1:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 10 + (load_shed) * 5)
                    else:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 10 + (load_shed) * 5)
                    reward11t = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 5)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3)

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + (
                                #load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 1e1)
                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (load_shed) * 1e3)
                    #reward1 = -(sum(powerflow2[0]["gen"][:, 1]) +(load_shedding_s+load_shedding_e)*1e3+(load_shed)*1e1)#R1

                    reward2 = (load_shedding_s + load_shedding_e + penalty) * 1 #+ aaa
                    reward2_t = (load_shedding_s + load_shedding_e + penalty) * 1#e2


                #reward2 = -(load_shedding_s + load_shedding_e) * 1e5 - (load_s11 + load_e11) * 10
                reward=reward11
                #reward = reward11 + reward2 - c * 1e5

                record = reward,reward2, reward00, reward11, c, c1,c2, action, attack, load_shedding_s+load_shedding_e,load_shed,powerflow2[0]["success"], action[11],action[12],penalty,aaa,penalty1,sum(powerflow2[0]["gen"][:, 1]),sum(powerflow[0]["gen"][:, 1])

        return s_, reward, record, 0, reward2,reward2_t,reward11t,reward00t


class IEGS4():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 24  # vlotege 发电机功率，######load shedding
        self.observation_space =43#有功无功负荷

    def reset(self):
        maxPl=np.zeros(shape=(42,25))
        minPl = np.zeros(shape=(42, 25))
        maxPl1 = np.zeros(shape=(42, 25))
        minPl1 = np.ones(shape=(42, 25))
        rate= np.array([
            0.95, 0.841666667, 0.812, 0.8, 0.7833333333, 0.733333333, 0.75, 0.803333333, 0.866666667, 0.9183333333,
            1.066666667, 1.116666667, 1.166666667, 1.041666667, 1.016666667,
            1.03333333, 1.0516666667, 1.1308333333, 1.166666667, 1.1608333333, 1.16666667, 1.133333333, 1.066666667, 1, 1
        ])#每小时负荷比例
        parameters1 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*1.5
        parameters2 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*0.5
        #负荷节点上下限度
        data=np.array([2,3,4,7,8,10,11,12,14,15,16,17,18,19,20,21,23,24,26,29,30])-1 #负荷节点
        for i in range(25):
            maxPl[:, i] = rate[i] * parameters1
            minPl[:, i] = rate[i] * parameters2
        load_shedding=maxPl-minPl
        state = np.random.uniform(maxPl, minPl)
        state1 = np.random.uniform(maxPl1, minPl1)
        #state1=state1*(maxPl-minPl)
        max_action = np.array(
            [1.05, 1.05, 1.05, 1.05, 1.05, 160, 100, 110, 60, 80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1])
        #max_action = np.array([1.05,1.05,1.05,1.05,1.05,1.05,160,100,110,60,80,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])#前六个是发电机电压控制，7-11是PV发电机输出功率控制，后面1是PQ节点的load shedding
        #min_action = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        min_action = np.array(
            [0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #min_action = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0])
        #max_action=np.array([2,60])
        #min_action=np.array([1,30])
        action_data=max_action-min_action
        action_basic=min_action
        #load_shedding=[]

        #data=GasData()
        return state, action_data, data, action_basic,load_shedding,minPl

    def step(self, action, action_data,data,action_basic,state_,state,step,load_shedding,attack,minPl,mark):
        #state_shedding=np.zeros(shape=(42))
        #state=state*load_shedding[:,step]+minPl[:,step]
        #state_=state_*load_shedding[:,step+1]+minPl[:,step+1]
        s_ = state_
        action = (action + 1) / 2
        #attack1 = (attack1 + 1) / 2*20
        mark1=np.array(
            [5,7,9,10,11,12,13,14,15,16,17,18,19,20])-1
        action = action * action_data + action_basic




        if attack!=-1:


            #attack =round(attack1[0])
            if attack==0:
                attack=7
            elif attack==1:
                attack=8
            elif attack==2:
                attack=9
            elif attack==3:
                attack=12
            elif attack==4:
                attack=16
            elif attack==5:
                attack=17
            elif attack==6:
                attack=18
            elif attack==7:
                attack=19
            elif attack==8:
                attack=20
            elif attack==9:
                attack=21
            elif attack==10:
                attack=22
            elif attack==11:
                attack=23
            elif attack==12:
                attack=24
            elif attack==13:
                attack=25
            elif attack==14:
                attack=28#28
            elif attack==15:
                attack=28
            elif attack==16:
                attack=31
            elif attack==17:
                attack=33
            elif attack==18:
                attack=37
            elif attack==19:
                attack=38
            elif attack==20:
                attack=39

                #if attack[i]==5:
                #    attack[i]=7

                ####################
                #if attack[i]==0:
                #    attack[i]=13
                #if attack[i]==29:
                #    attack[i]=30
                #if attack[i]==15:
                #    attack[i]=14
                #if attack[i]==35:
                #    attack[i]=12
                #if attack[i]==10:
                #    attack[i]=33
                #if attack[i]==40:
                #    attack[i]=12
                #if attack[i]==34:
                #    attack[i]=33
###############################################

            #for N-K attack
            ###################for i in range(len(attack)):
            #    mark=Find(attack[i],attack,i)
            #    if mark!=[]:
            #        p=1
            #        for j in range(len(mark)):
            #            attack[mark[j]]=attack[mark[j]]-p
            #########################            p+=1
        c=0
        c1=0

        load_shedding_s = 0
        penalty = 0
        shedding = np.zeros(shape=(21))


        #if load_shedding!=[]:
        #    for i in range(21):
        #        shedding[i]=load_shedding[i,step]*action[11]
        #    state_shedding[0:21] = state[0:21] - shedding
        #    state_shedding[22:42] = state[22:42]
        #    powerflow2 = runpf(case30(action, state_shedding, data, attack,0,0,0,[]))

        powerflow= runpf(case31(action, state, data,[],0,0,0,[],mark1))

        load_shed=0
        j=0
        for i in mark1:  # for MA and AA
            q = 0
            for p in data:
                if p == data[i]:
                    break
                q += 1
            load_shed += action[j + 10] * state[q]
            j+=1


        if powerflow[0]["success"]==False :
            reward=-1e4
            reward2= np.random.uniform(100, 99.99)
            record = reward, c1, action, attack
            reward2_t=reward2
            reward11t=reward
            reward00t = reward
        else:
            if attack!=-1:
                if attack==12 or attack==33:
                    a=1
                powerflow2 = runpf(case31(action, state, data, attack,[],[],[],[],mark1))

                for i in range(1):
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 1] - powerflow2[0]["gen"][i, 8])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 9] - powerflow2[0]["gen"][i, 1]))) / (
                                       (powerflow2[0]["gen"][i, 8]) - powerflow2[0]["gen"][i, 9])) ** 2
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / (
                                       (powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c1 += ((SolveMax(0,
                                     (powerflow2[0]["gen"][i + 1, 2] - powerflow2[0]["gen"][i + 1, 3])) + SolveMax(
                        0, (
                                powerflow2[0]["gen"][i + 1, 4] - powerflow2[0]["gen"][i + 1, 2]))) / (
                                   (powerflow2[0]["gen"][i + 1, 3]) - powerflow2[0]["gen"][i + 1, 4])) ** 2
                for i in range(30):
                    c1 += ((SolveMax(0, powerflow2[0]["bus"][i, 7] - 1.05) + SolveMax(0,
                                                                                      0.95 - powerflow2[0]["bus"][
                                                                                          i, 7])) / 0.1) ** 2
                # c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c1 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c1 = (c1 / 71) ** 0.5
                sou=int(powerflow2[0]["branch"][int(attack), 0])-1
                load_s = powerflow2[0]["bus"][sou, 2]
                end=int(powerflow2[0]["branch"][int(attack), 1])-1
                load_e = powerflow2[0]["bus"][end, 2]
                load_shedding_s = 0
                load_shedding_e = 0
                load_ss1 = 0
                load_ss2 = 0
                load_ee1 = 0
                load_ee2 = 0
                load_ss = 0
                load_ee = 0
                if sou==26 or end==26 or attack == 21 or attack == 24 or attack == 22 or attack== 23 or attack == 18 or attack == 25:
                    if attack == 21 or attack == 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:
                        if attack == 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                    if sou == 26 or end == 26:
                        powerflow2 = runpf(case31(action, state, data, attack, powerflow2[0]["bus"][28, 2]/2, powerflow2[0]["bus"][29, 2]/2, 28, 29,mark1))
                        load_shedding_s = powerflow2[0]["bus"][28, 2]/2
                        load_shedding_e = powerflow2[0]["bus"][29, 2]/2
                else:
                    for i in range(41):
                        mark111=Find(sou+1,powerflow2[0]["branch"][:, 0],int(attack))#以sou为入口
                        mark2=Find(end+1,powerflow2[0]["branch"][:, 1],int(attack))#以end为出口
                        mark11 = Find(sou+1, powerflow2[0]["branch"][:, 1], int(attack))#以sou为出口
                        mark22 = Find(end+1, powerflow2[0]["branch"][:, 0], int(attack))#以end为入口
                    if mark111!=[]:
                        for i in range(len(mark111)):
                            if powerflow2[0]["branch"][mark111[i], 5]<-powerflow2[0]["branch"][mark111[i], 13]:
                                load_ss1= 0.9*(sum(powerflow2[0]["branch"][mark111[j], 5] for j in range(len(mark111))))
                            else:
                                load_ss1 =0
                    if mark22 != []:
                        for i in range(len(mark22)):
                            if powerflow2[0]["branch"][mark22[i], 5]<-powerflow2[0]["branch"][mark22[i], 15]:
                                load_ee1= 0.9*(sum(powerflow2[0]["branch"][mark22[j], 5] for j in range(len(mark22))))
                            else:
                                load_ee1 =0
                    if mark11 != []:
                        for i in range(len(mark11)):
                            if powerflow2[0]["branch"][mark11[i], 5]<-powerflow2[0]["branch"][mark11[i], 15]:
                                load_ss2= 0.9*(sum(powerflow2[0]["branch"][mark11[j], 5] for j in range(len(mark11))))
                            else:
                                load_ss2 =0
                    if mark2 != []:
                        for i in range(len(mark2)):
                            if powerflow2[0]["branch"][mark2[i], 5]<-powerflow2[0]["branch"][mark2[i], 13]:
                                load_ee2= 0.9*(sum(powerflow2[0]["branch"][mark2[j], 5] for j in range(len(mark2))))
                            else:
                                load_ee2 =0
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss1 =0
                        load_ss2 = 0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee1 = 0
                        load_ee2 = 0
                    if load_ss1!=0 or load_ss2!=0 or load_ee1!=0 or load_ee2!=0:
                        if mark111!=[]:
                            load_ss += load_ss1/ len(mark111)
                        if mark11 != []:
                            load_ss += load_ss2 / len(mark11)
                        if mark22!=[]:
                            load_ee += load_ee1/ len(mark22)
                        if mark2 != []:
                            load_ee += load_ee2 / len(mark2)

                        #load_ss=(load_ss1+load_ss2)/(len(mark1)+len(mark11))
                        #load_ee = (load_ee1 + load_ee2) / (len(mark2) + len(mark22))
                    else:
                        load_ss =load_s
                        load_ee=load_e
                    #if load_ss==0:
                    #    load_ss=0.8*load_s
                    #if load_ee==0:
                    #    load_ee=0.8*load_e
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss =0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee = 0
                    if powerflow2[0]["success"] == True:

                        load_shedding_s = load_s - load_ss
                        load_shedding_e = load_e - load_ee
                        if load_shedding_s < 0:
                            if sou!=1 and sou!=23 and sou!=0 and sou!=22:
                                penalty=-load_shedding_s
                                #penalty=0
                                load_ss= load_s
                                load_shedding_s = 0
                            if sou == 1:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][6, 2] + 0.2 * powerflow2[0]["bus"][7, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.8 * powerflow2[0]["bus"][6, 2], 0.8 * powerflow2[0]["bus"][7, 2], 6, 7,mark1))
                            elif sou == 23:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][21, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.2 * powerflow2[0]["bus"][21, 2], 0, 21, -1,mark1))
                            elif sou == 22:
                                #action1[9] = (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty+=(action[8]-(load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8]))
                                action[8]= (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            elif sou == 0:
                                #action1[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]) / action_data[6] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty +=(action[5]-(load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5]))
                                action[5] = (load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_s = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        elif load_shedding_e < 0:
                            if end != 22:
                                penalty=-load_shedding_e
                                #penalty=0
                                load_ee = load_e
                                load_shedding_e=0
                            if end == 22:
                                #action1[9] = (load_e + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_e = 0
                                #penalty = 0
                                penalty += (action[8] - (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8]))
                                action[8] = (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_e = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        else:
                            if attack==12 or attack==33:
                                a=1
                            else:
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))

                            # if end==26:
                            #    action1[8]=(load_e+powerflow2[0]["branch"][mark2[0], 5]-action_basic)/action_data*20-10
                            #a=1
                            #if load_ss>load_s:
                            #    load_shedding_s=0
                            #if load_ee>load_e:
                            #    load_shedding_e=0

                    if attack == 21 or attack== 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:

                        if attack== 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack== 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack== 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                if attack==12 :
                    load_shedding_s=powerflow[0]["bus"][10, 2]
                    load_shedding_e = 0
                    penalty=0
                if attack==33 :
                    load_shedding_s=1
                    load_shedding_e = powerflow[0]["bus"][25, 2]
                    penalty=0
            for i in range(1):#slack节点有功功率和无功功率是否超出限额
                c+=((SolveMax(0,(powerflow[0]["gen"][i, 1]-powerflow[0]["gen"][i, 8]))+SolveMax(0,(powerflow[0]["gen"][i, 9]-powerflow[0]["gen"][i, 1])))/((powerflow[0]["gen"][i, 8])-powerflow[0]["gen"][i, 9]))**2
                c += ((SolveMax(0, (powerflow[0]["gen"][i, 2] - powerflow[0]["gen"][i, 3])) + SolveMax(0, (powerflow[0]["gen"][i, 4] - powerflow[0]["gen"][i, 2]))) / ((powerflow[0]["gen"][i, 3]) - powerflow[0]["gen"][i, 4])) ** 2
            for i in range(5):#PV节点无功功率是否超出限额
                c += ((SolveMax(0, (powerflow[0]["gen"][i+1, 2] - powerflow[0]["gen"][i+1, 3])) + SolveMax(0, (
                            powerflow[0]["gen"][i+1, 4] - powerflow[0]["gen"][i+1, 2]))) / (
                                  (powerflow[0]["gen"][i+1, 3]) - powerflow[0]["gen"][i+1, 4])) ** 2
            for i in range(30):#节点电压是否超出限额
                c+=((SolveMax(0,powerflow[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow[0]["bus"][i,7]))/0.1)**2
            #c=(c/31)**0.5
            for i in range(41):#线路流量是否超出限额
                c+=(SolveMax(0,powerflow[0]["branch"][i,13]-powerflow[0]["branch"][i,5])/powerflow[0]["branch"][i,5])**2
            c = (c / 72) ** 0.5


            ###
            c2=0

            if attack!=-1:
                for i in range(1):
                    c2+=((SolveMax(0,(powerflow2[0]["gen"][i, 1]-powerflow2[0]["gen"][i, 8]))+SolveMax(0,(powerflow2[0]["gen"][i, 9]-powerflow2[0]["gen"][i, 1])))/((powerflow2[0]["gen"][i, 8])-powerflow2[0]["gen"][i, 9]))**2
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / ((powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i+1, 2] - powerflow2[0]["gen"][i+1, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i+1, 4] - powerflow2[0]["gen"][i+1, 2]))) / (
                                      (powerflow2[0]["gen"][i+1, 3]) - powerflow2[0]["gen"][i+1, 4])) ** 2
                for i in range(30):
                    c2+=((SolveMax(0,powerflow2[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow2[0]["bus"][i,7]))/0.1)**2
                #c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c2 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c2 = (c2 / 71) ** 0.5
            if c1==0:
                load_shedding_s+=c2*1e5

            ###


            if powerflow[0]["success"]==False :
                reward0=-1e8
                reward00 = -1e8
            else:
                reward00 = -(sum(powerflow[0]["gen"][:, 1])+c*1e3)
                reward0 = -(sum(powerflow[0]["gen"][:, 1]) )#R0
            aaa = np.random.uniform(1000, 999.99)
            if attack==-1:
                reward=reward00
                reward2=0
                reward1=[]
                record = reward, reward0, reward1, c, action
                reward00t=0
                reward2_t=0
                reward11t=0
            else:
                if powerflow2[0]["success"]==False :
                    if mark==1:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 5 + 10000)
                    else:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3+ (load_shed) * 5 + 2000)
                    reward11t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 + (load_shed) * 5 + 2000)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3 )
                    #reward1=-1e6
                    #reward11 = -1e6

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1])  + (load_shed) * 1e2)
                    reward2 =10
                    reward2_t =20
                else:
                    #if load_shedding_s + load_shedding_e + penalty>20:
                    #    load_shedding_s = 0
                    #    load_shedding_e = 0
                    #    penalty = 20
                    if mark==1:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 8 + (load_shed) * 5)
                    else:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 8 + (load_shed) * 5)
                    reward11t = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e3 + (
                                load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 5)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 1e3)

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + (
                                #load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 1e1)
                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (load_shed) * 1e3)
                    #reward1 = -(sum(powerflow2[0]["gen"][:, 1]) +(load_shedding_s+load_shedding_e)*1e3+(load_shed)*1e1)#R1

                    reward2 = (load_shedding_s + load_shedding_e + penalty) * 1 #+ aaa
                    reward2_t = (load_shedding_s + load_shedding_e + penalty) * 1#e2


                #reward2 = -(load_shedding_s + load_shedding_e) * 1e5 - (load_s11 + load_e11) * 10
                reward=reward11
                #reward = reward11 + reward2 - c * 1e5

                record = reward,reward2, reward00, reward11, c, c1,c2, action, attack, load_shedding_s+load_shedding_e,load_shed,powerflow2[0]["success"], action[11],action[12],penalty,aaa,sum(powerflow2[0]["gen"][:, 1]),sum(powerflow[0]["gen"][:, 1])

        return s_, reward, record, 0, reward2+1e-5,reward2_t,reward11t,reward00t


class IEGS5():
    def __init__(self):
        #self.action_space = 32#vlotege 发电机功率，######load shedding
        self.action_space = 24  # vlotege 发电机功率，######load shedding
        self.observation_space =43#有功无功负荷

    def reset(self):
        maxPl=np.zeros(shape=(42,25))
        minPl = np.zeros(shape=(42, 25))
        maxPl1 = np.zeros(shape=(42, 25))
        minPl1 = np.ones(shape=(42, 25))
        rate= np.array([
             1, 1.041666667, 0.912, 0.85, 0.833333333, 0.733333333, 0.75, 1.083333333, 1.466666667, 1.883333333, 1.966666667, 2.216666667, 2.066666667, 1.941666667, 2.116666667,
        2.233333333, 2.316666667, 2.208333333, 2, 2, 1.833333333, 1.633333333, 1.433333333,1,1
        ])#每小时负荷比例
        parameters1 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*1.5
        parameters2 = np.array([21.7, 2.4, 7.6, 22.8, 30, 5.8, 2.4, 11.2, 12.2, 8.2, 13.5, 9, 13.2, 9.5, 12.2, 17.5, 3.2, 10.7, 3.5, 2.4, 10.6,
                                12.7, 1.2, 1.6, 10.9, 30, 2, 1.2, 7.5, 1.6, 2.5, 1.8, 5.8, 0.9, 3.4, 0.7, 11.2, 1.6, 6.7, 2.3, 0.9, 1.9])#*0.5
        #负荷节点上下限度
        data=np.array([2,3,4,7,8,10,11,12,14,15,16,17,18,19,20,21,23,24,26,29,30])-1 #负荷节点
        for i in range(25):
            maxPl[:, i] = rate[i] * parameters1
            minPl[:, i] = rate[i] * parameters2
        load_shedding=maxPl-minPl
        state = np.random.uniform(maxPl, minPl)
        state1 = np.random.uniform(maxPl1, minPl1)
        #state1=state1*(maxPl-minPl)
        max_action = np.array(
            [1.05, 1.05, 1.05, 1.05, 1.05, 160, 100, 110, 60, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #max_action = np.array([1.05,1.05,1.05,1.05,1.05,1.05,160,100,110,60,80,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])#前六个是发电机电压控制，7-11是PV发电机输出功率控制，后面1是PQ节点的load shedding
        #min_action = np.array([0.95,0.95,0.95,0.95,0.95,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        min_action = np.array(
            [0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0])
        #min_action = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 60, 60, 60, 60, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0])
        #max_action=np.array([2,60])
        #min_action=np.array([1,30])
        action_data=max_action-min_action
        action_basic=min_action
        #load_shedding=[]

        #data=GasData()
        return state, action_data, data, action_basic,load_shedding,minPl

    def step(self, action, action_data,data,action_basic,state_,state,step,load_shedding,attack,minPl,mark):
        #state_shedding=np.zeros(shape=(42))
        #state=state*load_shedding[:,step]+minPl[:,step]
        #state_=state_*load_shedding[:,step+1]+minPl[:,step+1]
        s_ = state_
        action = (action + 1) / 2
        mark1=np.array(
            [5,7,9,10,11,12,13,14,15,16,17,18,19,20])-1
        action = action * action_data + action_basic





        if attack!=-1:


            attack =attack
            if attack==0:
                attack=7
            elif attack==1:
                attack=8
            elif attack==2:
                attack=9
            elif attack==3:
                attack=12
            elif attack==4:
                attack=16
            elif attack==5:
                attack=17
            elif attack==6:
                attack=18
            elif attack==7:
                attack=19
            elif attack==8:
                attack=20
            elif attack==9:
                attack=21
            elif attack==10:
                attack=22
            elif attack==11:
                attack=23
            elif attack==12:
                attack=24
            elif attack==13:
                attack=25
            elif attack==14:
                attack=28#28
            elif attack==15:
                attack=28
            elif attack==16:
                attack=31
            elif attack==17:
                attack=33
            elif attack==18:
                attack=37
            elif attack==19:
                attack=38
            elif attack==20:
                attack=39

                #if attack[i]==5:
                #    attack[i]=7

                ####################
                #if attack[i]==0:
                #    attack[i]=13
                #if attack[i]==29:
                #    attack[i]=30
                #if attack[i]==15:
                #    attack[i]=14
                #if attack[i]==35:
                #    attack[i]=12
                #if attack[i]==10:
                #    attack[i]=33
                #if attack[i]==40:
                #    attack[i]=12
                #if attack[i]==34:
                #    attack[i]=33
###############################################

            #for N-K attack
            ###################for i in range(len(attack)):
            #    mark=Find(attack[i],attack,i)
            #    if mark!=[]:
            #        p=1
            #        for j in range(len(mark)):
            #            attack[mark[j]]=attack[mark[j]]-p
            #########################            p+=1
        c=0
        c1=0

        load_shedding_s = 0
        penalty = 0
        shedding = np.zeros(shape=(21))


        #if load_shedding!=[]:
        #    for i in range(21):
        #        shedding[i]=load_shedding[i,step]*action[11]
        #    state_shedding[0:21] = state[0:21] - shedding
        #    state_shedding[22:42] = state[22:42]
        #    powerflow2 = runpf(case30(action, state_shedding, data, attack,0,0,0,[]))

        powerflow= runpf(case31(action, state, data,[],0,0,0,[],mark1))

        load_shed=0
        j=0
        for i in mark1:  # for MA and AA
            q = 0
            for p in data:
                if p == data[i]:
                    break
                q += 1
            load_shed += action[j + 10] * state[q]
            j+=1


        if powerflow[0]["success"]==False :
            reward=-1e4
            reward2= np.random.uniform(100, 99.99)
            record = reward, c1, action, attack
            reward2_t=reward2
            reward11t=reward
            reward00t = reward
        else:
            if attack!=-1:
                if attack==12 or attack==33:
                    a=1
                powerflow2 = runpf(case31(action, state, data, attack,[],[],[],[],mark1))

                for i in range(1):
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 1] - powerflow2[0]["gen"][i, 8])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 9] - powerflow2[0]["gen"][i, 1]))) / (
                                       (powerflow2[0]["gen"][i, 8]) - powerflow2[0]["gen"][i, 9])) ** 2
                    c1 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / (
                                       (powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c1 += ((SolveMax(0,
                                     (powerflow2[0]["gen"][i + 1, 2] - powerflow2[0]["gen"][i + 1, 3])) + SolveMax(
                        0, (
                                powerflow2[0]["gen"][i + 1, 4] - powerflow2[0]["gen"][i + 1, 2]))) / (
                                   (powerflow2[0]["gen"][i + 1, 3]) - powerflow2[0]["gen"][i + 1, 4])) ** 2
                for i in range(30):
                    c1 += ((SolveMax(0, powerflow2[0]["bus"][i, 7] - 1.05) + SolveMax(0,
                                                                                      0.95 - powerflow2[0]["bus"][
                                                                                          i, 7])) / 0.1) ** 2
                # c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c1 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c1 = (c1 / 71) ** 0.5
                sou=int(powerflow2[0]["branch"][int(attack), 0])-1
                load_s = powerflow2[0]["bus"][sou, 2]
                end=int(powerflow2[0]["branch"][int(attack), 1])-1
                load_e = powerflow2[0]["bus"][end, 2]
                load_shedding_s = 0
                load_shedding_e = 0
                load_ss1 = 0
                load_ss2 = 0
                load_ee1 = 0
                load_ee2 = 0
                load_ss = 0
                load_ee = 0
                if sou==26 or end==26 or attack == 21 or attack == 24 or attack == 22 or attack== 23 or attack == 18 or attack == 25:
                    if attack == 21 or attack == 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:
                        if attack == 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack == 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack == 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                    if sou == 26 or end == 26:
                        powerflow2 = runpf(case31(action, state, data, attack, powerflow2[0]["bus"][28, 2]/2, powerflow2[0]["bus"][29, 2]/2, 28, 29,mark1))
                        load_shedding_s = powerflow2[0]["bus"][28, 2]/2
                        load_shedding_e = powerflow2[0]["bus"][29, 2]/2
                else:
                    for i in range(41):
                        mark111=Find(sou+1,powerflow2[0]["branch"][:, 0],int(attack))#以sou为入口
                        mark2=Find(end+1,powerflow2[0]["branch"][:, 1],int(attack))#以end为出口
                        mark11 = Find(sou+1, powerflow2[0]["branch"][:, 1], int(attack))#以sou为出口
                        mark22 = Find(end+1, powerflow2[0]["branch"][:, 0], int(attack))#以end为入口
                    if mark111!=[]:
                        for i in range(len(mark111)):
                            if powerflow2[0]["branch"][mark111[i], 5]<-powerflow2[0]["branch"][mark111[i], 13]:
                                load_ss1= 0.9*(sum(powerflow2[0]["branch"][mark111[j], 5] for j in range(len(mark111))))
                            else:
                                load_ss1 =0
                    if mark22 != []:
                        for i in range(len(mark22)):
                            if powerflow2[0]["branch"][mark22[i], 5]<-powerflow2[0]["branch"][mark22[i], 15]:
                                load_ee1= 0.9*(sum(powerflow2[0]["branch"][mark22[j], 5] for j in range(len(mark22))))
                            else:
                                load_ee1 =0
                    if mark11 != []:
                        for i in range(len(mark11)):
                            if powerflow2[0]["branch"][mark11[i], 5]<-powerflow2[0]["branch"][mark11[i], 15]:
                                load_ss2= 0.9*(sum(powerflow2[0]["branch"][mark11[j], 5] for j in range(len(mark11))))
                            else:
                                load_ss2 =0
                    if mark2 != []:
                        for i in range(len(mark2)):
                            if powerflow2[0]["branch"][mark2[i], 5]<-powerflow2[0]["branch"][mark2[i], 13]:
                                load_ee2= 0.9*(sum(powerflow2[0]["branch"][mark2[j], 5] for j in range(len(mark2))))
                            else:
                                load_ee2 =0
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss1 =0
                        load_ss2 = 0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee1 = 0
                        load_ee2 = 0
                    if load_ss1!=0 or load_ss2!=0 or load_ee1!=0 or load_ee2!=0:
                        if mark111!=[]:
                            load_ss += load_ss1/ len(mark111)
                        if mark11 != []:
                            load_ss += load_ss2 / len(mark11)
                        if mark22!=[]:
                            load_ee += load_ee1/ len(mark22)
                        if mark2 != []:
                            load_ee += load_ee2 / len(mark2)

                        #load_ss=(load_ss1+load_ss2)/(len(mark1)+len(mark11))
                        #load_ee = (load_ee1 + load_ee2) / (len(mark2) + len(mark22))
                    else:
                        load_ss =load_s
                        load_ee=load_e
                    #if load_ss==0:
                    #    load_ss=0.8*load_s
                    #if load_ee==0:
                    #    load_ee=0.8*load_e
                    if powerflow2[0]["bus"][sou, 2]==0:
                        load_ss =0
                    if powerflow2[0]["bus"][end, 2] == 0:
                        load_ee = 0
                    if powerflow2[0]["success"] == True:

                        load_shedding_s = load_s - load_ss
                        load_shedding_e = load_e - load_ee
                        if load_shedding_s < 0:
                            if sou!=1 and sou!=23 and sou!=0 and sou!=22:
                                penalty=-load_shedding_s
                                #penalty=0
                                load_ss= load_s
                                load_shedding_s = 0
                            if sou == 1:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][6, 2] + 0.2 * powerflow2[0]["bus"][7, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.8 * powerflow2[0]["bus"][6, 2], 0.8 * powerflow2[0]["bus"][7, 2], 6, 7,mark1))
                            elif sou == 23:
                                load_shedding_s = 0.2 * powerflow2[0]["bus"][21, 2]
                                powerflow2 = runpf(case31(action, state, data, attack, 0.2 * powerflow2[0]["bus"][21, 2], 0, 21, -1,mark1))
                            elif sou == 22:
                                #action1[9] = (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty+=(action[8]-(load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8]))
                                action[8]= (load_s + powerflow2[0]["branch"][mark11[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            elif sou == 0:
                                #action1[6] = (load_s + powerflow2[0]["branch"][mark1[0], 5] - action_basic[6]) / action_data[6] * 20 - 10

                                load_shedding_s = 0
                                #penalty = 0
                                penalty +=(action[5]-(load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5]))
                                action[5] = (load_s + powerflow2[0]["branch"][mark111[0], 5] - action_basic[5])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_s = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        elif load_shedding_e < 0:
                            if end != 22:
                                penalty=-load_shedding_e
                                #penalty=0
                                load_ee = load_e
                                load_shedding_e=0
                            if end == 22:
                                #action1[9] = (load_e + powerflow2[0]["branch"][mark22[0], 5] - action_basic[9]) / action_data[9] * 20 - 10

                                load_shedding_e = 0
                                #penalty = 0
                                penalty += (action[8] - (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8]))
                                action[8] = (load_s + powerflow2[0]["branch"][mark22[0], 5] - action_basic[8])
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                            else:
                                load_shedding_e = 0
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))
                        else:
                            if attack==12 or attack==33:
                                a=1
                            else:
                                powerflow2 = runpf(case31(action, state, data, attack, load_ss, load_ee, sou, end,mark1))

                            # if end==26:
                            #    action1[8]=(load_e+powerflow2[0]["branch"][mark2[0], 5]-action_basic)/action_data*20-10
                            #a=1
                            #if load_ss>load_s:
                            #    load_shedding_s=0
                            #if load_ee>load_e:
                            #    load_shedding_e=0

                    if attack == 21 or attack== 24 or attack == 22 or attack == 23 or attack == 18 or attack == 25:

                        if attack== 21 or attack == 24:
                            load_shedding_s = 3 / 4 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2] +
                                                       powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 0,mark1))
                        if attack == 22:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][18, 2] + powerflow2[0]["bus"][19, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 1,mark1))
                        if attack== 23:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][17, 2] + powerflow2[0]["bus"][18, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 2,mark1))
                        if attack== 18:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][15, 2] + powerflow2[0]["bus"][16, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 3,mark1))
                        if attack == 25:
                            load_shedding_s = 2 / 3 * (powerflow2[0]["bus"][14, 2] + powerflow2[0]["bus"][15, 2])
                            powerflow2 = runpf(case31(action, state, data, attack, 0, 0, 0, 4,mark1))
                if attack==12 :
                    load_shedding_s=powerflow[0]["bus"][10, 2]
                    load_shedding_e = 0
                    penalty=0
                if attack==33 :
                    load_shedding_s=1
                    load_shedding_e = powerflow[0]["bus"][25, 2]
                    penalty=0
            for i in range(1):#slack节点有功功率和无功功率是否超出限额
                c+=((SolveMax(0,(powerflow[0]["gen"][i, 1]-powerflow[0]["gen"][i, 8]))+SolveMax(0,(powerflow[0]["gen"][i, 9]-powerflow[0]["gen"][i, 1])))/((powerflow[0]["gen"][i, 8])-powerflow[0]["gen"][i, 9]))**2
                c += ((SolveMax(0, (powerflow[0]["gen"][i, 2] - powerflow[0]["gen"][i, 3])) + SolveMax(0, (powerflow[0]["gen"][i, 4] - powerflow[0]["gen"][i, 2]))) / ((powerflow[0]["gen"][i, 3]) - powerflow[0]["gen"][i, 4])) ** 2
            for i in range(5):#PV节点无功功率是否超出限额
                c += ((SolveMax(0, (powerflow[0]["gen"][i+1, 2] - powerflow[0]["gen"][i+1, 3])) + SolveMax(0, (
                            powerflow[0]["gen"][i+1, 4] - powerflow[0]["gen"][i+1, 2]))) / (
                                  (powerflow[0]["gen"][i+1, 3]) - powerflow[0]["gen"][i+1, 4])) ** 2
            for i in range(30):#节点电压是否超出限额
                c+=((SolveMax(0,powerflow[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow[0]["bus"][i,7]))/0.1)**2
            #c=(c/31)**0.5
            for i in range(41):#线路流量是否超出限额
                c+=(SolveMax(0,powerflow[0]["branch"][i,13]-powerflow[0]["branch"][i,5])/powerflow[0]["branch"][i,5])**2
            c = (c / 72) ** 0.5


            ###
            c2=0

            if attack!=-1:
                for i in range(1):
                    c2+=((SolveMax(0,(powerflow2[0]["gen"][i, 1]-powerflow2[0]["gen"][i, 8]))+SolveMax(0,(powerflow2[0]["gen"][i, 9]-powerflow2[0]["gen"][i, 1])))/((powerflow2[0]["gen"][i, 8])-powerflow2[0]["gen"][i, 9]))**2
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i, 2] - powerflow2[0]["gen"][i, 3])) + SolveMax(0, (powerflow2[0]["gen"][i, 4] - powerflow2[0]["gen"][i, 2]))) / ((powerflow2[0]["gen"][i, 3]) - powerflow2[0]["gen"][i, 4])) ** 2
                for i in range(5):
                    c2 += ((SolveMax(0, (powerflow2[0]["gen"][i+1, 2] - powerflow2[0]["gen"][i+1, 3])) + SolveMax(0, (
                                powerflow2[0]["gen"][i+1, 4] - powerflow2[0]["gen"][i+1, 2]))) / (
                                      (powerflow2[0]["gen"][i+1, 3]) - powerflow2[0]["gen"][i+1, 4])) ** 2
                for i in range(30):
                    c2+=((SolveMax(0,powerflow2[0]["bus"][i,7]-1.05)+SolveMax(0,0.95-powerflow2[0]["bus"][i,7]))/0.1)**2
                #c1=(c1/31)**0.5
                for i in range(41):  # 线路流量是否超出限额
                    c2 += (SolveMax(0, powerflow2[0]["branch"][i, 13] - powerflow2[0]["branch"][i, 5]) /
                          powerflow2[0]["branch"][i, 5]) ** 2
                c2 = (c2 / 71) ** 0.5
            if c1==0:
                load_shedding_s+=c2*1e5

            ###


            if powerflow[0]["success"]==False :
                reward0=-1e8
                reward00 = -1e8
            else:
                reward00 = -(sum(powerflow[0]["gen"][:, 1])+c*1e3)
                reward0 = -(sum(powerflow[0]["gen"][:, 1]) )#R0
            aaa = np.random.uniform(1000, 999.99)
            if attack==-1:
                reward=reward00
                reward2=0
                reward1=[]
                record = reward, reward0, reward1, c, action
                reward00t=0
                reward2_t=0
                reward11t=0
            else:
                if powerflow2[0]["success"]==False :
                    if mark==1:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3 + (load_shed) * 5 + 10000)
                    else:
                        reward11 = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3+ (load_shed) * 5 + 2000)
                    reward11t = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3 + (load_shed) * 5 + 2000)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3 )
                    #reward1=-1e6
                    #reward11 = -1e6

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1])  + (load_shed) * 1e2)
                    reward2 =10
                    reward2_t =20
                else:
                    #if load_shedding_s + load_shedding_e + penalty>20:
                    #    load_shedding_s = 0
                    #    load_shedding_e = 0
                    #    penalty = 20
                    if mark==1:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 5e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 5)
                    else:
                        reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 5e3 + (
                                    load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 5)
                    reward11t = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 5e3 + (
                                load_shedding_s + load_shedding_e + penalty) * 1e1 + (load_shed) * 5)
                    reward00t = -(sum(powerflow[0]["gen"][:, 1]) + c * 5e3)

                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + (
                                #load_shedding_s + load_shedding_e + penalty) * 1e2 + (load_shed) * 1e1)
                    #reward11 = -(sum(powerflow2[0]["gen"][:, 1]) + c2 * 1e4 + (load_shed) * 1e3)
                    #reward1 = -(sum(powerflow2[0]["gen"][:, 1]) +(load_shedding_s+load_shedding_e)*1e3+(load_shed)*1e1)#R1

                    reward2 = (load_shedding_s + load_shedding_e + penalty) * 1 #+ aaa
                    reward2_t = (load_shedding_s + load_shedding_e + penalty) * 1#e2


                #reward2 = -(load_shedding_s + load_shedding_e) * 1e5 - (load_s11 + load_e11) * 10
                reward=reward11
                #reward = reward11 + reward2 - c * 1e5

                record = reward,reward2, reward00, reward11, c, c1,c2, action, attack, load_shedding_s+load_shedding_e,load_shed,powerflow2[0]["success"], action[11],action[12],penalty,aaa,sum(powerflow2[0]["gen"][:, 1]),sum(powerflow[0]["gen"][:, 1])

        return s_, reward, record, 0, reward2,reward2_t,reward11t,reward00t

def SolveMax(data1,data2):
    data=0
    if data2>data1:
        data=data2
    if data2<=data1:
        data=data1
    return data
def Find(data,array,j):
    mark = []
    for i in range(len(array)):
        if i!=j:
            if data==array[i]:
                mark.append(i)
    return mark


