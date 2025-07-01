# coding=gbk
import math
import random

import numpy as np
import hypernetx as hnx
from matplotlib import pyplot as plt
from scipy.stats import norm


class InfoSpreading(object):
    def __init__(self):
        self.nodesNum = 0
        self.alpha = 0.4
        self.beta = 0.3
        self.gamma1 = 0.1
        self.eta = 0.4
        self.xi = 0.2
        self.lamda = 1
        self.W = 0.5
        self.T = 0
        self.kexi = 0
        self.timeStep = 50
        self.repeatNum = 10
        self.state = np.zeros((self.nodesNum, 2), int)
        self.imageDataSEIRA = np.zeros((self.timeStep, 25), float)
        self.imageDataSIS = np.zeros((self.timeStep, 2), float)
        self.imageDataSIR = np.zeros((self.timeStep, 3), float)
        self.imageDataSEIR = np.zeros((self.timeStep, 4), float)
        self.scenes = {}
        self.degreeList = []
        self.imgSave = "ImgSave/"
        self.importPath = "networkModel/"

    def funRunSIS(self):
        self.alpha = 0.7
        self.gamma1 = 0.3
        self.importModel(999,1,2,1)
        self.sprSIS("avgHi")
        self.imgDrawSIS()

    def funRunSIR(self):
        self.alpha = 0.5
        self.gamma1 = 0.3
        self.importModel(999,1,2,1)
        self.sprSIR("avgHi")
        self.imgDrawSIR()

    def funRunSEIR(self):
        self.alpha = 0.5
        self.beta = 0.3
        self.eta = 0.3
        self.gamma1 = 0.3
        self.xi = 0.3
        self.importModel(999, 1, 2, 1)
        self.sprSEIR("avgHi")
        self.imgDrawSEIR()

    def funRunSEEIAR2(self):
        self.SEEIARbeta1 = 0.6
        self.SEEIARbeta2 = 0.3
        self.SEEIARalpha1 = 0.8
        self.SEEIARalpha2 = 0.2
        self.SEEIARalpha3 = 0.4
        self.SEEIARgama = 0.01
        self.SEEIARlamda1 = 0.2
        self.SEEIARlamda2 = 0.2
        self.SEEIARlamda3 = 0.1
        self.SEEIARlamda4 = 0.1
        self.importModel(999, 1, 2, 1)
        self.sprSEEIAR2("max-avgHi",0)
        self.imgDrawSEEIAR()

    def funRunSEEIARModel(self):
        self.lamda = 0.1
        self.W = 0.1
        self.T = 0
        self.importModel(999,1,2,1)
        self.pardataNew(0.3,0.1**2)
        self.sprSEEIAR("max-avgHi",0)
        self.imgDrawSEEIAR()

    def pardataNew(self,voi1,voi2):
        self.H = hnx.Hypergraph(self.scenes)
        self.degreeList = hnx.degree_dist(self.H)
        self.VOI = self.VOIfun(voi1, voi2)

    def ffun(self,i,j):
        sumjdegree = 0
        for node in self.H.neighbors('v'+str(j)):
            sumjdegree += self.degreeList[int(node[1:])-1]
        f = self.degreeList[i-1]/sumjdegree
        return f

    def Rwfun(self,i,j):
        return self.ffun(i,j)/(self.ffun(i,j)+self.ffun(j,i))/2

    def ufun(self,W,t,T):
        if t < T:
            return 1
        else:
            u = pow(math.e,-(W*(t-T)))
            return u

    def VOIfun(self,mu,sigma):
        norm_dist = norm(loc=mu, scale=sigma)
        random_samples = norm_dist.rvs(size=self.nodesNum)
        return random_samples

    def beta1beta2(self,i,j,t):
        return self.kexi + (1-self.kexi)*self.Rwfun(i,j)

    def SEEIARgamma(self,i,j,t):
        gamma = self.VOI[i-1]*self.Rwfun(i,j)*math.e**(-self.lamda*t)
        if gamma < 1:
            return gamma/3
        else:
            return 1

    def alpha1(self,i,t):
        if t < self.T:
            return (self.kexi + (1-self.kexi) * self.VOI[i-1])*3
        else:
            return (self.kexi + (1-self.kexi) * self.VOI[i - 1] * self.ufun(self.W, t - 1, self.T))*3

    def alpha2(self,i,t):
        return (self.kexi + (1-self.kexi) * math.e**(-self.lamda*t))/3

    def alpha3(self,i,t):
        return self.kexi + (1-self.kexi) * self.VOI[i-1]*3

    def lamda12(self,t):
        if t < self.T:
            return (1-pow(math.e,-(self.lamda*t)))/3
        else:
            return (1-(pow(math.e,-(self.lamda*t)) * self.ufun(self.W, t-1, self.T)))/3

    def lamda34(self,t):
        return (1-pow(math.e,-(self.lamda*t)))/3

    def importModel(self, N, m1, m2, m):
        self.nodesNum = N
        self.state = np.zeros((self.nodesNum, 2), int)
        self.scenes = {}
        print("import n" + str(N) + "m1" + str(m1) + "m2" + str(m2) + "m" + str(m) + " model")
        self.incidenceMatrix = np.genfromtxt(
            self.importPath + "n" + str(self.nodesNum) + "m1" + str(m1) + "m2" + str(m2) + "m" + str(m) + "Inc.txt", delimiter=' ')
        for i in range(len(self.incidenceMatrix[0])):
            eTuple = ()
            for j in range(self.nodesNum):
                if self.incidenceMatrix[j][i] != 0:
                    eTuple += ('v' + str(j + 1),)
            self.scenes["E" + str(i + 1)] = eTuple

    def sprSIS(self, select):
        selectnode1 = 0
        for repeat in range(self.repeatNum):
            SNodes = []
            INodes = []
            print("\rRepeat experiment progress:" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            self.state = np.zeros((self.nodesNum, 2), int)
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0
                SNodes.append(i + 1)
            if select == "avgHi":
                selectnode1 = self.avgHi()
            self.state[selectnode1 - 1][1] = 1
            SNodes.remove(selectnode1)
            INodes.append(selectnode1)

            for t in range(self.timeStep):
                ISNodesChange = []
                SINodesChange = []
                edgeFlagNum = []
                if t >= 1:
                    for i in INodes:
                        if random.randint(1, 100) <= 100 * self.gamma1:
                            ISNodesChange.append(i)
                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 1:
                            if 'E' + str(e + 1) not in edgeFlagNum:
                                edgeFlagNum.append('E' + str(e + 1))
                                for i in self.scenes['E' + str(e + 1)]:
                                    if self.state[int(i[1:]) - 1][1] == 0:
                                        if random.randint(1, 100) <= 100 * self.alpha:
                                            SINodesChange.append(int(i[1:]))
                ISNodesChange = list(set(ISNodesChange))
                SINodesChange = list(set(SINodesChange))
                for i in ISNodesChange:
                    self.state[i - 1][1] = 0
                    SNodes.append(i)
                    INodes.remove(i)
                for i in SINodesChange:
                    self.state[i - 1][1] = 1
                    SNodes.remove(i)
                    INodes.append(i)
                self.imageDataSIS[t][0] += len(SNodes) / self.nodesNum / self.repeatNum
                self.imageDataSIS[t][1] += len(INodes) / self.nodesNum / self.repeatNum
        print()

    def sprSIR(self, select):
        selectnode1 = 0
        for repeat in range(self.repeatNum):
            SNodes = []
            INodes = []
            RNodes = []
            print("\rRepeat experiment progress:" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            self.state = np.zeros((self.nodesNum, 2), int)
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0
                SNodes.append(i + 1)
            if select == "avgHi":
                selectnode1 = self.avgHi()
            self.state[selectnode1 - 1][1] = 1
            SNodes.remove(selectnode1)
            INodes.append(selectnode1)

            for t in range(self.timeStep):
                SNodesChange = []
                INodesChange = []
                RNodesChange = []
                if t >= 1:
                    for i in INodes:
                        if random.randint(1, 100) <= 100 * self.gamma1:
                            RNodesChange.append(i)
                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 1:
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:
                                    if random.randint(1, 100) <= 100 * self.alpha:
                                        SNodesChange.append(int(i[1:]))
                                        INodesChange.append(int(i[1:]))
                RNodesChange = list(set(RNodesChange))
                SNodesChange = list(set(SNodesChange))
                INodesChange = list(set(INodesChange))
                for i in RNodesChange:
                    self.state[i - 1][1] = 2
                    RNodes.append(i)
                    INodes.remove(i)
                for i in SNodesChange:
                    SNodes.remove(i)
                for i in INodesChange:
                    self.state[i - 1][1] = 1
                    INodes.append(i)
                self.imageDataSIR[t][0] += len(SNodes) / self.nodesNum / self.repeatNum
                self.imageDataSIR[t][1] += len(INodes) / self.nodesNum / self.repeatNum
                self.imageDataSIR[t][2] += len(RNodes) / self.nodesNum / self.repeatNum
        print()

    def sprSEIR(self, select):
        selectnode1 = 0
        for repeat in range(self.repeatNum):
            SNodes = []
            ENodes = []
            INodes = []
            RNodes = []
            print("\rRepeat experiment progress:" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            self.state = np.zeros((self.nodesNum, 2), int)
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0
                SNodes.append(i + 1)
            if select == "avgHi":
                selectnode1 = self.avgHi()
            self.state[selectnode1 - 1][1] = 2
            SNodes.remove(selectnode1)
            INodes.append(selectnode1)

            for t in range(self.timeStep):
                NodesChange = []
                if t >= 1:
                    for i in INodes:
                        if random.randint(1, 100) <= 100 * self.gamma1:
                            if i not in NodesChange:
                                NodesChange.append(i)
                                self.state[i - 1][1] = 3
                                RNodes.append(i)
                                INodes.remove(i)
                    for i in ENodes:
                        if random.randint(1, 100) <= 100 * self.xi:
                            if i not in NodesChange:
                                NodesChange.append(i)
                                self.state[i - 1][1] = 3
                                RNodes.append(i)
                                ENodes.remove(i)
                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 2:
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.alpha:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 1
                                            SNodes.remove(int(i[1:]))
                                            ENodes.append(int(i[1:]))
                                    elif ran <= 100 * (self.alpha + self.beta):
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 2
                                            SNodes.remove(int(i[1:]))
                                            INodes.append(int(i[1:]))
                                elif self.state[int(i[1:]) - 1][1] == 1:
                                    if random.randint(1, 100) <= 100 * self.eta:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 2
                                            ENodes.remove(int(i[1:]))
                                            INodes.append(int(i[1:]))

                self.imageDataSEIR[t][0] += len(SNodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIR[t][1] += len(ENodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIR[t][2] += len(INodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIR[t][3] += len(RNodes) / self.nodesNum / self.repeatNum
        print()

    def sprSEEIAR(self, select, nu):
        selectnode1 = 0
        selectnode2 = 0
        for repeat in range(self.repeatNum):
            SNodes = []
            E1Nodes = []
            E2Nodes = []
            INodes = []
            ANodes = []
            RNodes = []
            print("\rRepeat experiment progress:" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            self.state = np.zeros((self.nodesNum, 2), int)
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0
                SNodes.append(i + 1)
            if repeat == 0:
                if select == "avg-maxHi":
                    selectnode1 = self.avgHi()
                    selectnode2 = self.maxHi()
                elif select == "avg-minHi":
                    selectnode1 = self.avgHi()
                    selectnode2 = self.minHi()
                elif select == "max-avgHi":
                    selectnode1 = self.maxHi()
                    selectnode2 = self.avgHi()
                elif select == "min-avgHi":
                    selectnode1 = self.minHi()
                    selectnode2 = self.avgHi()

            self.state[selectnode1 - 1][1] = 2
            self.state[selectnode2 - 1][1] = 4
            SNodes.remove(selectnode1)
            SNodes.remove(selectnode2)
            INodes.append(selectnode1)
            ANodes.append(selectnode2)

            for t in range(self.timeStep):
                NodesChange = []
                if t >= 1:
                    for i in INodes:
                        if random.randint(1, 100) <= 100 * self.lamda12(t):
                            if i not in NodesChange:
                                NodesChange.append(i)
                                self.state[i - 1][1] = 3
                                RNodes.append(i)
                                INodes.remove(i)
                    for i in ANodes:
                        if random.randint(1, 100) <= 100 * self.lamda12(t):
                            if i not in NodesChange:
                                NodesChange.append(i)
                                self.state[i - 1][1] = 3
                                RNodes.append(i)
                                ANodes.remove(i)

                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 2:
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.beta1beta2(int(node[1:]), int(i[1:]), t)*2:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 1
                                            SNodes.remove(int(i[1:]))
                                            E1Nodes.append(int(i[1:]))

                        elif self.state[int(node[1:]) - 1][1] == 4:
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.beta1beta2(int(node[1:]), int(i[1:]), t)/2:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 5
                                            SNodes.remove(int(i[1:]))
                                            E2Nodes.append(int(i[1:]))
                                elif self.state[int(i[1:]) - 1][1] == 2:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.SEEIARgamma(int(node[1:]), int(i[1:]),t):
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 4
                                            INodes.remove(int(i[1:]))
                                            ANodes.append(int(i[1:]))
                for i in E1Nodes:
                    if i not in NodesChange:
                        ran = random.randint(1, 100)
                        if ran <= 100 * (self.alpha1(i, t)/(self.alpha1(i, t) + self.alpha2(i,t) + self.lamda34(t))):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 2
                            INodes.append(i)
                            E1Nodes.remove(i)
                        elif ran <= 100 * (self.alpha1(i, t) + self.alpha2(i,t)/(self.alpha1(i, t) + self.alpha2(i,t) + self.lamda34(t))):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 4
                            ANodes.append(i)
                            E1Nodes.remove(i)
                        else:
                            NodesChange.append(i)
                            self.state[i - 1][1] = 3
                            RNodes.append(i)
                            E1Nodes.remove(i)
                for i in E2Nodes:
                    if i not in NodesChange:
                        ran = random.randint(1, 100)
                        if ran <= 100 * self.alpha3(i,t):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 4
                            ANodes.append(i)
                            E2Nodes.remove(i)
                        elif ran <= 100 * (self.alpha3(i,t) + self.lamda34(t)):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 3
                            RNodes.append(i)
                            E2Nodes.remove(i)
                self.imageDataSEIRA[t][0 + 6 * nu] += len(SNodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][1 + 6 * nu] += len(E1Nodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][2 + 6 * nu] += len(E2Nodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][3 + 6 * nu] += len(INodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][4 + 6 * nu] += len(ANodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][5 + 6 * nu] += len(RNodes) / self.nodesNum / self.repeatNum
        print()

    def sprSEEIAR2(self, select, nu):
        selectnode1 = 0
        selectnode2 = 0
        for repeat in range(self.repeatNum):
            SNodes = []
            E1Nodes = []
            E2Nodes = []
            INodes = []
            ANodes = []
            RNodes = []
            print("\rRepeat experiment progress:" + str(repeat + 1) + "/" + str(self.repeatNum), end="")
            self.state = np.zeros((self.nodesNum, 2), int)
            for i in range(self.nodesNum):
                self.state[i][0] = i + 1
                self.state[i][1] = 0
                SNodes.append(i + 1)
            if repeat == 0:
                if select == "avg-maxHi":
                    selectnode1 = self.avgHi()
                    selectnode2 = self.maxHi()
                elif select == "avg-minHi":
                    selectnode1 = self.avgHi()
                    selectnode2 = self.minHi()
                elif select == "max-minHi":
                    selectnode2 = self.minHi()
                    selectnode1 = self.maxHi()
                elif select == "max-avgHi":
                    selectnode1 = self.maxHi()
                    selectnode2 = self.avgHi()
                elif select == "min-avgHi":
                    selectnode1 = self.minHi()
                    selectnode2 = self.avgHi()

            self.state[selectnode1 - 1][1] = 2
            self.state[selectnode2 - 1][1] = 4
            SNodes.remove(selectnode1)
            SNodes.remove(selectnode2)
            INodes.append(selectnode1)
            ANodes.append(selectnode2)

            for t in range(self.timeStep):
                NodesChange = []
                if t >= 1:
                    for i in INodes:
                        if random.randint(1, 100) <= 100 * self.SEEIARlamda1:
                            if i not in NodesChange:
                                NodesChange.append(i)
                                self.state[i - 1][1] = 3
                                RNodes.append(i)
                                INodes.remove(i)
                    for i in ANodes:
                        if random.randint(1, 100) <= 100 * self.SEEIARlamda2:
                            if i not in NodesChange:
                                NodesChange.append(i)
                                self.state[i - 1][1] = 3
                                RNodes.append(i)
                                ANodes.remove(i)
                for e in range(len(self.incidenceMatrix[0])):
                    for node in self.scenes['E' + str(e + 1)]:
                        if self.state[int(node[1:]) - 1][1] == 2:
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.SEEIARbeta1:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 1
                                            SNodes.remove(int(i[1:]))
                                            E1Nodes.append(int(i[1:]))

                        elif self.state[int(node[1:]) - 1][1] == 4:
                            for i in self.scenes['E' + str(e + 1)]:
                                if self.state[int(i[1:]) - 1][1] == 0:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.SEEIARbeta2:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 5
                                            SNodes.remove(int(i[1:]))
                                            E2Nodes.append(int(i[1:]))
                                elif self.state[int(i[1:]) - 1][1] == 2:
                                    ran = random.randint(1, 100)
                                    if ran <= 100 * self.SEEIARgama:
                                        if int(i[1:]) not in NodesChange:
                                            NodesChange.append(int(i[1:]))
                                            self.state[int(i[1:]) - 1][1] = 4
                                            INodes.remove(int(i[1:]))
                                            ANodes.append(int(i[1:]))
                for i in E1Nodes:
                    if i not in NodesChange:
                        ran = random.randint(1, 100)
                        if ran <= 100 * self.SEEIARalpha1:
                            NodesChange.append(i)
                            self.state[i - 1][1] = 2
                            INodes.append(i)
                            E1Nodes.remove(i)
                        elif ran <= 100 * (self.SEEIARalpha1 + self.SEEIARalpha2):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 4
                            ANodes.append(i)
                            E1Nodes.remove(i)
                        elif ran <= 100 * (self.SEEIARalpha1 + self.SEEIARalpha2 + self.SEEIARlamda3):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 3
                            RNodes.append(i)
                            E1Nodes.remove(i)
                for i in E2Nodes:
                    if i not in NodesChange:
                        ran = random.randint(1, 100)
                        if ran <= 100 * self.SEEIARalpha3:
                            NodesChange.append(i)
                            self.state[i - 1][1] = 4
                            ANodes.append(i)
                            E2Nodes.remove(i)
                        elif ran <= 100 * (self.SEEIARalpha3 + self.SEEIARlamda4):
                            NodesChange.append(i)
                            self.state[i - 1][1] = 3
                            RNodes.append(i)
                            E2Nodes.remove(i)
                self.imageDataSEIRA[t][0 + 6 * nu] += len(SNodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][1 + 6 * nu] += len(E1Nodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][2 + 6 * nu] += len(E2Nodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][3 + 6 * nu] += len(INodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][4 + 6 * nu] += len(ANodes) / self.nodesNum / self.repeatNum
                self.imageDataSEIRA[t][5 + 6 * nu] += len(RNodes) / self.nodesNum / self.repeatNum
        print()

    def imgDrawSEEIAR(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [1]
        yvalue1 = [0]
        yvalue2 = [0]
        yvalue3 = [0]
        yvalue4 = [0]
        yvalue5 = [0]
        for i in range(1, self.timeStep):
            yvalue0.append(self.imageDataSEIRA[i][0])
            yvalue1.append(self.imageDataSEIRA[i][1])
            yvalue2.append(self.imageDataSEIRA[i][2])
            yvalue3.append(self.imageDataSEIRA[i][3])
            yvalue4.append(self.imageDataSEIRA[i][4])
            yvalue5.append(self.imageDataSEIRA[i][5])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#99CCFF', linestyle='-', linewidth=1, marker='s', markersize=8)
        ax.plot(t, yvalue1, color='#FFCC99', linestyle='-', linewidth=1, marker='d', markersize=8)
        ax.plot(t, yvalue2, color='#C0C0C0', linestyle='-', linewidth=1, marker='p', markersize=8)
        ax.plot(t, yvalue3, color='#66CC00', linestyle='-', linewidth=1, marker='^', markersize=8)
        ax.plot(t, yvalue4, color='#FF9999', linestyle='-', linewidth=1, marker='o', markersize=8)
        ax.plot(t, yvalue5, color='#E5CCFF', linestyle='-', linewidth=1, marker='X', markersize=8)
        ax.legend(labels=[r"S", r"E$_1$", r"E$_2$", r"I", r"A", r"R"], ncol=1,
                  fontsize=19)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.savefig(self.imgSave + "SEIAR.svg", format='svg', dpi=600)
        plt.show()

    def imgDrawSIS(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [1]
        yvalue1 = [0]
        for i in range(1, self.timeStep):
            yvalue0.append(self.imageDataSIS[i][0])
            yvalue1.append(self.imageDataSIS[i][1])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#99CCFF', linestyle='-', linewidth=1, marker='s', markersize=8)
        ax.plot(t, yvalue1, color='#66CC00', linestyle='-', linewidth=1, marker='p', markersize=8)
        ax.legend(labels=["S", "I"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.savefig(self.imgSave + "SIS.svg", format='svg', dpi=600)
        plt.show()

    def imgDrawSIR(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [1]
        yvalue1 = [0]
        yvalue2 = [0]
        for i in range(1, self.timeStep):
            yvalue0.append(self.imageDataSIR[i][0])
            yvalue1.append(self.imageDataSIR[i][1])
            yvalue2.append(self.imageDataSIR[i][2])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#99CCFF', linestyle='-', linewidth=1, marker='s', markersize=8)
        ax.plot(t, yvalue1, color='#66CC00', linestyle='-', linewidth=1, marker='p', markersize=8)
        ax.plot(t, yvalue2, color='#E5CCFF', linestyle='-', linewidth=1, marker='o', markersize=8)
        ax.legend(labels=["S", "I", "R"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.savefig(self.imgSave + "SIR.svg", format='svg', dpi=600)
        plt.show()

    def imgDrawSEIR(self):
        t = [i for i in range(self.timeStep)]
        yvalue0 = [1]
        yvalue1 = [0]
        yvalue2 = [0]
        yvalue3 = [0]
        for i in range(1, self.timeStep):
            yvalue0.append(self.imageDataSEIR[i][0])
            yvalue1.append(self.imageDataSEIR[i][1])
            yvalue2.append(self.imageDataSEIR[i][2])
            yvalue3.append(self.imageDataSEIR[i][3])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(t, yvalue0, color='#99CCFF', linestyle='-', linewidth=1, marker='s', markersize=8)
        ax.plot(t, yvalue1, color='#FFCC99', linestyle='-', linewidth=1, marker='d', markersize=8)
        ax.plot(t, yvalue2, color='#66CC00', linestyle='-', linewidth=1, marker='p', markersize=8)
        ax.plot(t, yvalue3, color='#E5CCFF', linestyle='-', linewidth=1, marker='o', markersize=8)
        ax.legend(labels=["S", "E", "I", "R"], ncol=1, fontsize=20)
        plt.xlabel("t", fontsize=25)
        plt.ylabel("ρ", fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(width=1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        plt.savefig(self.imgSave + "SEIR.svg", format='svg', dpi=600)
        plt.show()

    def avgHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        avg = (max(degreeList) + min(degreeList)) // 2
        while avg not in degreeList:
            avg -= 1
        selectnode = degreeList.index(avg) + 1
        return selectnode

    def maxHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        selectnode = degreeList.index(max(degreeList)) + 1
        return selectnode

    def minHi(self):
        H = hnx.Hypergraph(self.scenes)
        degreeList = hnx.degree_dist(H)
        selectnode = degreeList.index(min(degreeList)) + 1
        return selectnode

if __name__ == '__main__':
    infospr = InfoSpreading()
    # infospr.funRunSIS()
    # infospr.funRunSIR()
    # infospr.funRunSEIR()
    # infospr.funRunSEEIAR2()
    infospr.funRunSEEIARModel()
