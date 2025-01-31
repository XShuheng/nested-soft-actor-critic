# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 30 bus, 6 generator case.
"""

from numpy import array

def case31(action,state,data,attack,load_ss,load_ee,sou,end,mark1):
    """Power flow data for 30 bus, 6 generator case.
    Please see L{caseformat} for details on the case file format.

    Based on data from ...

    Alsac, O. & Stott, B., I{"Optimal Load Flow with Steady State Security"},
    IEEE Transactions on Power Apparatus and Systems, Vol. PAS 93, No. 3,
    1974, pp. 745-751.

    ... with branch parameters rounded to nearest 0.01, shunt values divided
    by 100 and shunt on bus 10 moved to bus 5, load at bus 5 zeroed out.
    Generator locations, costs and limits and bus areas were taken from ...

    Ferrero, R.W., Shahidehpour, S.M., Ramesh, V.C., I{"Transaction analysis
    in deregulated power systems using game theory"}, IEEE Transactions on
    Power Systems, Vol. 12, No. 3, Aug 1997, pp. 1340-1347.

    Generator Q limits were derived from Alsac & Stott, using their Pmax
    capacities. V limits and line |S| limits taken from Alsac & Stott.

    @return: Power flow data for 30 bus, 6 generator case.
    @see: U{http://www.pserc.cornell.edu/matpower/}

    """


    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [2, 2, 21.7, 12.7, 0, 0, 1, 1, 0, 135, 1, 1.1, 0.95],
        [3, 1, 2.4, 1.2, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [4, 1, 7.6, 1.6, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [5, 1, 0, 0, 0, 0.19, 1, 1, 0, 135, 1, 1.05, 0.95],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [7, 1, 22.8, 10.9, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [8, 1, 30, 30, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [9, 1, 0, 0, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [10, 1, 5.8, 2, 0, 0, 3, 1, 0, 135, 1, 1.05, 0.95],
        [11, 1, 2.4, 0, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [12, 1, 11.2, 7.5, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [13, 2, 0, 0, 0, 0, 2, 1, 0, 135, 1, 1.1, 0.95],
        [14, 1, 6.2, 1.6, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [15, 1, 8.2, 2.5, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [16, 1, 3.5, 1.8, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [17, 1, 9, 5.8, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [18, 1, 3.2, 0.9, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [19, 1, 9.5, 3.4, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [20, 1, 2.2, 0.7, 0, 0, 2, 1, 0, 135, 1, 1.05, 0.95],
        [21, 1, 17.5, 11.2, 0, 0, 3, 1, 0, 135, 1, 1.05, 0.95],
        [22, 2, 0, 0, 0, 0, 3, 1, 0, 135, 1, 1.1, 0.95],
        [23, 2, 3.2, 1.6, 0, 0, 2, 1, 0, 135, 1, 1.1, 0.95],
        [24, 1, 8.7, 6.7, 0, 0.04, 3, 1, 0, 135, 1, 1.05, 0.95],
        [25, 1, 0, 0, 0, 0, 3, 1, 0, 135, 1, 1.05, 0.95],
        [26, 1, 3.5, 2.3, 0, 0, 3, 1, 0, 135, 1, 1.05, 0.95],
        [27, 2, 0, 0, 0, 0, 3, 1, 0, 135, 1, 1.1, 0.95],
        [28, 1, 0, 0, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95],
        [29, 1, 2.4, 0.9, 0, 0, 3, 1, 0, 135, 1, 1.05, 0.95],
        [30, 1, 10.6, 1.9, 0, 0, 3, 1, 0, 135, 1, 1.05, 0.95]
    ])

    k = 0
    for i in data:  # for MA and AA
        ppc["bus"][i, 2] = state[k]
        ppc["bus"][i, 3] = state[21 + k]
        k += 1
    j=0
    for i in mark1:  #for MA and AA
        q=0
        for p in data:
            if p==data[i]:
                break
            q+=1
        ppc["bus"][data[i],2]=state[q]-action[j+10]*state[q]
        j+=1


    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 23.54, 0, 150, -20, 1.05, 100, 1, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 60.97, 0, 60, -20, 1, 100, 1, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [22, 21.59, 0, 62.5, -15, 1, 100, 1, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [27, 26.91, 0, 48.7, -15, 1, 100, 1, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [23, 19.2, 0, 40, -10, 1, 100, 1, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [13, 37, 0, 44.7, -15, 1, 100, 1, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    for i in range(5):
        ppc["gen"][i+1, 5] = action[i]
    for i in range(5):
        ppc["gen"][i+1, 1] = action[i+5]


    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 2, 0.02, 0.06, 0.03, 130, 130, 130, 0, 0, 1, -360, 360],#1
        [1, 3, 0.05, 0.19, 0.02, 130, 130, 130, 0, 0, 1, -360, 360],#2
        [2, 4, 0.06, 0.17, 0.02, 65, 65, 65, 0, 0, 1, -360, 360],#3
        [3, 4, 0.01, 0.04, 0, 130, 130, 130, 0, 0, 1, -360, 360],#4
        [2, 5, 0.05, 0.2, 0.02, 130, 130, 130, 0, 0, 1, -360, 360],#5
        [2, 6, 0.06, 0.18, 0.02, 65, 65, 65, 0, 0, 1, -360, 360],#6
        [4, 6, 0.01, 0.04, 0, 90, 90, 90, 0, 0, 1, -360, 360],#7
        [5, 7, 0.05, 0.12, 0.01, 70, 70, 70, 0, 0, 1, -360, 360],#8
        [6, 7, 0.03, 0.08, 0.01, 130, 130, 130, 0, 0, 1, -360, 360],#9
        [6, 8, 0.01, 0.04, 0, 32, 32, 32, 0, 0, 1, -360, 360],#10
        [6, 9, 0, 0.21, 0, 65, 65, 65, 0, 0, 1, -360, 360],#11
        [6, 10, 0, 0.56, 0, 32, 32, 32, 0, 0, 1, -360, 360],#12
        [9, 11, 0, 0.21, 0, 65, 65, 65, 0, 0, 1, -360, 360],#13
        [9, 10, 0, 0.11, 0, 65, 65, 65, 0, 0, 1, -360, 360],#14
        [4, 12, 0, 0.26, 0, 65, 65, 65, 0, 0, 1, -360, 360],#15
        [12, 13, 0, 0.14, 0, 65, 65, 65, 0, 0, 1, -360, 360],#16
        [12, 14, 0.12, 0.26, 0, 32, 32, 32, 0, 0, 1, -360, 360],#17
        [12, 15, 0.07, 0.13, 0, 32, 32, 32, 0, 0, 1, -360, 360],#18
        [12, 16, 0.09, 0.2, 0, 32, 32, 32, 0, 0, 1, -360, 360],#19
        [14, 15, 0.22, 0.2, 0, 16, 16, 16, 0, 0, 1, -360, 360],#20
        [16, 17, 0.08, 0.19, 0, 16, 16, 16, 0, 0, 1, -360, 360],#21
        [15, 18, 0.11, 0.22, 0, 16, 16, 16, 0, 0, 1, -360, 360],#22
        [18, 19, 0.06, 0.13, 0, 16, 16, 16, 0, 0, 1, -360, 360],#23
        [19, 20, 0.03, 0.07, 0, 32, 32, 32, 0, 0, 1, -360, 360],#24
        [10, 20, 0.09, 0.21, 0, 32, 32, 32, 0, 0, 1, -360, 360],#25
        [10, 17, 0.03, 0.08, 0, 32, 32, 32, 0, 0, 1, -360, 360],#26
        [10, 21, 0.03, 0.07, 0, 32, 32, 32, 0, 0, 1, -360, 360],#27
        [10, 22, 0.07, 0.15, 0, 32, 32, 32, 0, 0, 1, -360, 360],#28
        [21, 22, 0.01, 0.02, 0, 32, 32, 32, 0, 0, 1, -360, 360],#29
        [15, 23, 0.1, 0.2, 0, 32, 32, 32, 0, 0, 1, -360, 360],##########################限额16换成32#30
        [22, 24, 0.12, 0.18, 0, 16, 16, 16, 0, 0, 1, -360, 360],#31
        [23, 24, 0.13, 0.27, 0, 16, 16, 16, 0, 0, 1, -360, 360],#32
        [24, 25, 0.19, 0.33, 0, 16, 16, 16, 0, 0, 1, -360, 360],#33
        [25, 26, 0.25, 0.38, 0, 16, 16, 16, 0, 0, 1, -360, 360],#34
        [25, 27, 0.11, 0.21, 0, 16, 16, 16, 0, 0, 1, -360, 360],#35
        #[28, 29, 0.11, 0.21, 0, 20, 20, 20, 0, 0, 1, -360, 360],###################
        #[25, 30, 0.11, 0.21, 0, 16, 16, 16, 0, 0, 1, -360, 360],###################
        [28, 27, 0, 0.4, 0, 65, 65, 65, 0, 0, 1, -360, 360],#36
        [27, 29, 0.22, 0.42, 0, 16, 16, 16, 0, 0, 1, -360, 360],#37
        [27, 30, 0.32, 0.6, 0, 16, 16, 16, 0, 0, 1, -360, 360],#38
        [29, 30, 0.24, 0.45, 0, 16, 16, 16, 0, 0, 1, -360, 360],#39
        [8, 28, 0.06, 0.2, 0.02, 32, 32, 32, 0, 0, 1, -360, 360],#40
        [6, 28, 0.02, 0.06, 0.01, 32, 32, 32, 0, 0, 1, -360, 360]#41
    ])
    ppc["branch"][:,2]=0.11
    ppc["branch"][:, 3] = 0.21
    if attack== 12 or attack==33:
        if attack==12:
            ppc["bus"][10, 2] = 0
        if attack==33:
            ppc["bus"][25, 2] = 0
    else:


        ppc["branch"][attack, 10] = 0




            #else:
            #    ppc["gen"][int(attack[0]) +1, 7] = 0
    if load_ss!=0:
        ppc["bus"][sou,2]=load_ss
    if load_ee!=0:
        ppc["bus"][end,2]=load_ee
    if sou==0:
        if end==0:
            ppc["bus"][17,2]=3/4*ppc["bus"][17,2]
            ppc["bus"][18, 2] = 3 / 4 * ppc["bus"][18, 2]
            ppc["bus"][19, 2] = 3 / 4 * ppc["bus"][19, 2]
        if end==1:
            ppc["bus"][18, 2] = 2 / 3 * ppc["bus"][18, 2]
            ppc["bus"][19, 2] = 2 / 3 * ppc["bus"][19, 2]
        if end ==2:
            ppc["bus"][17, 2] = 2 / 3 * ppc["bus"][17, 2]
            ppc["bus"][18, 2] = 2 / 3 * ppc["bus"][18, 2]
        if end ==3:
            ppc["bus"][15, 2] = 2 / 3 * ppc["bus"][15, 2]
            ppc["bus"][16, 2] = 2 / 3 * ppc["bus"][16, 2]
        if end ==4:
            ppc["bus"][14, 2] = 2 / 3 * ppc["bus"][14, 2]
            ppc["bus"][15, 2] = 2 / 3 * ppc["bus"][15, 2]
    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 8],
        [2, 23],
        [3, 26],
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 3, 0.02, 2, 0],
        [2, 0, 0, 3, 0.0175, 1.75, 0],
        [2, 0, 0, 3, 0.0625, 1, 0],
        [2, 0, 0, 3, 0.00834, 3.25, 0],
        [2, 0, 0, 3, 0.025, 3, 0],
        [2, 0, 0, 3, 0.025, 3, 0]
    ])

    return ppc
