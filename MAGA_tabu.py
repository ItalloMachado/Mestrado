# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 02:25:12 2020

@author: Itallo
"""

import math
import random
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import csv
from pgmpy.readwrite import BIFReader
import time
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import statistics
import gc


Bnl = importr('bnlearn')
rscore=robjects.r['score']


def BIC_r(G,rdados,auxn):
    """Função que calcula o BIC utilizando uma biblioteca do R
    Entrada: G= Grafo
             rdados= dados no formato da biblioteca do R
             auxn= Variável que calcula quantas vezes foi calculado a métrica BIC
    Saída:   score= valor do BIC do grafo G
             auxn= Variável que calcula quantas vezes foi calculado a métrica BIC"""
    auxn+=1
    string_bn=""
    for i in G.nodes():
        string_bn+="["+i
        aux=0
        for j in G.predecessors(i):
            if aux==0:
                string_bn+="|"
                aux+=1
            else:
                string_bn+=":"
            string_bn+=j
        string_bn+="]"
    
    dag = Bnl.model2network(string_bn)
    score=float(rscore(dag,rdados).r_repr())
    
    if auxn%100==0:   # limpando a memoria em cada 100 avaliações do BIC
        robjects.r('gc()')

    return score,auxn

def srt_network(G,rdados,auxn):
    """Função que transforma uma rede em string
    Entrada: G= Grafo
             rdados= dados no formato da biblioteca do R
             auxn= Variável que calcula quantas vezes foi calculado a métrica BIC
    Saída: string_bn= string referente ao grafo G"""
    auxn+=1
    string_bn=""
    for i in G.nodes():
        string_bn+="["+i
        aux=0
        for j in G.predecessors(i):
            if aux==0:
                string_bn+="|"
                aux+=1
            else:
                string_bn+=":"
            string_bn+=j
        string_bn+="]"
    return string_bn

def hamming2(target,ind1,nodes):
    TC=0
    TE=len(target.edges())
    IE=0
    for i in target.edges():
        if ind1.has_edge(i[0],i[1]):
            TC+=1
        if ind1.has_edge(i[1],i[0]):
            IE+=1
    SLF=TC/TE
    TLF=(TC+IE)/TE

    return SLF,TLF
    
def F1Score(target,ind1,nodes):
    """Função que calcula a métrica F1
    Entrada: target=Grafo desejado
             ind1 = Grafo que será comparado
             nodes= vetor com os nomes dos nós do problema
    Saída: valores da métrica F1"""
    
    TP=0 #True positive
    FN=0 #false negative
    FP=0 #false positive
    TN=0 #true negative
    for i in nodes:
        for j in nodes:           
            if i!=j:
                if target.has_edge(i,j):
                    if ind1.has_edge(i,j):
                        TP+=1
                    else:
                        FN+=1
                else:
                    if ind1.has_edge(i,j):
                        FP+=1
                    else:
                        TN+=1

    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=(TP+TN)/(TP+TN+FN+FP)
    f1score=2*(recall*precision)/(recall+precision)
    return f1score, accuracy,precision,recall

def evaluateScore(x, n):
    score = 0
    for i in range(n):
        score = score - x[i] * math.sin(math.sqrt(abs(x[i])))
    return score

def FindNeighbors(i, j, Lsize):
    if i == 0:
        i1 = Lsize - 1
    else:
        i1 = i - 1
    if i == Lsize - 1:
        i2 = 0
    else:
        i2 = i + 1
    if j == 0:
        j1 = Lsize - 1
    else:
        j1 = j - 1
    if j == Lsize - 1:
        j2 = 0
    else:
        j2 = j + 1
    return [i1, j1, i2, j2]

class agent:
    def __init__(self, variable, i, j, Lsize, rdados):
        aux2=0
        self.variable = variable
        temp = time.time()
        [scr,aux2]=BIC_r(variable,rdados,aux2)
        self.score = abs(scr)
        temp = time.time()-temp
        self.tempo=temp
        self.pos = [i * (Lsize) + j]
        aux = FindNeighbors(i, j, Lsize)
        self.neighbors = [aux[0] * (Lsize) + j, i * (Lsize) + aux[1], aux[2] * (Lsize) + j, i * (Lsize) + aux[3]]
        
    def newvariables(self, val, scr):
        self.variable = val
        self.score = scr

def createAgents(Lsize, rdados,nodes,aux_bic):
    agents = []
    cont=0

    for i in range(Lsize):
        for j in range(Lsize):
            G=nx.fast_gnp_random_graph(len(nodes),0.1,directed=True)
            x = nx.DiGraph([(u,v,{'weight':1}) for (u,v) in G.edges() if u<v])
            random.shuffle(nodes)
            mapping={}
            random.shuffle(nodes)
            for k in range(len(nodes)):
                mapping.update({k:nodes[k]})
            x=nx.relabel_nodes(x, mapping)
            for a in nodes:
                if a not in x.nodes():
                    x.add_node(a)
            agents.append(agent(x, i, j, Lsize, rdados))
            aux_bic+=1
            cont+=1
    return agents,aux_bic


def s_createAgents(Lsize, L, rdata,nodes):
    agents = []
    for i in range(Lsize):
        for j in range(Lsize):
            if i == 0 and j == 0:
                agents.append(agent(L, i, j, Lsize, rdata))
            else:
                [L_aux,ns]=Mutation(L,nodes)
                agents.append(agent(L_aux, i, j, Lsize, rdata))
    return agents

def search_dag(G,edge_a,edge_b):
    no_dag=list(nx.simple_cycles(G))
    while no_dag != []:
        no_dag=no_dag[0]
        if len(no_dag)>2:
            rand_i=random.randint(0,len(no_dag)-1)
            if rand_i == 0:
                rand_aux=rand_i+1
            elif rand_i == len(no_dag)-1:
                rand_aux=rand_i-1
            else:
                rand_aux=random.random()
                if rand_aux<=0.5:
                    rand_aux=rand_i+1
                else:
                    rand_aux=rand_i-1
            aux=0
            while (no_dag[rand_i]==edge_a and no_dag[rand_aux]==edge_b) and aux<10:
                aux+=1
                if rand_i==0:
                    rand_i=rand_aux+1
                elif rand_aux == len(no_dag)-1:
                    rand_aux=rand_i-1
                else:
                    if random.random()<0.5:
                        rand_i=rand_aux+1
                    else:
                        rand_aux=rand_i-1
            if aux<10:     
                if rand_i<rand_aux:
                    if random.random()<=0.5:
                      G.remove_edge(no_dag[rand_i], no_dag[rand_aux]) 
                    else:
                      G.remove_edge(no_dag[rand_i], no_dag[rand_aux])
                      G.add_edge(no_dag[rand_aux],no_dag[rand_i])
                else:
                    if random.random()<=0.5:
                      G.remove_edge(no_dag[rand_aux],no_dag[rand_i]) 
                    else:
                      G.remove_edge(no_dag[rand_aux], no_dag[rand_i])
                      G.add_edge(no_dag[rand_i],no_dag[rand_aux])
            else:
               G.remove_edge(no_dag[rand_i], no_dag[rand_aux])  
        else:
            G.remove_edge(edge_b,edge_a)
        no_dag=list(nx.simple_cycles(G))     
    return G


def Crossover(ind1,ind2,nodes,aux_bic,goalscore):
    a=ind1.copy()
    a.update(ind2.copy())
    f1 = nx.DiGraph()
    f2 = nx.DiGraph()
    for i in nodes:
        if i not in f1.nodes():
            f1.add_node(i)
        if i not in f2.nodes():
            f2.add_node(i)
    for i in ind1.edges():
        if i in ind2.edges():
            f1.add_edge(i[0],i[1])
            f2.add_edge(i[0],i[1])
    for i in a.edges():
        if  not f1.has_edge(i[0],i[1]):
            if f1.has_edge(i[1],i[0]):
               r=random.random()
               if r<=0.5:
                  f1.remove_edge(i[1],i[0])
                  f1.add_edge(i[0],i[1])
                  search_dag(f1,i[0],i[1])
               else:
                  if f2.has_edge(i[1],i[0]): 
                      f2.remove_edge(i[1],i[0])
                      f2.add_edge(i[0],i[1])
                      search_dag(f2,i[0],i[1])
            else:
                r=random.random()
                if r<=0.5:
                    f1.add_edge(i[0],i[1])
                    search_dag(f1,i[0],i[1])
                else:
                    f2.add_edge(i[0],i[1])
                    search_dag(f2,i[0],i[1])

    [score_f1,aux_bic]=BIC_r(f1,rdados,aux_bic)
    [score_f2,aux_bic]=BIC_r(f2,rdados,aux_bic)
    score_f1=abs(score_f1)
    score_f2=abs(score_f2)
    if score_f1<score_f2:
        return f1,score_f1,aux_bic
    else:
        return f2,score_f2,aux_bic

def Mutation(ind1,nodes):
    R = nx.DiGraph()
    R=ind1.copy()
    action="NNN"
    for a in nodes:
        if a not in R.nodes():
            R.add_node(a)

    node1= random.randint(0,len(nodes)-1)
    node2= node1
    while(node1 == node2):
        node2= random.randint(0,len(nodes)-1)
    rand = random.random()
    if ind1.has_edge(nodes[node1],nodes[node2]):
        if rand <=0.5:
            R.remove_edge(nodes[node1],nodes[node2])
            R.add_edge(nodes[node2],nodes[node1])
            search_dag(R,node1,node2)
            action="right"
        else:
            R.remove_edge(nodes[node1],nodes[node2])
            action="remove"
    elif ind1.has_edge(nodes[node2],nodes[node1]):
        if rand <=0.5:
            R.remove_edge(nodes[node2],nodes[node1])
            R.add_edge(nodes[node1],nodes[node2])
            search_dag(R,node1,node2)
            action="left"
        else:
            R.remove_edge(nodes[node2],nodes[node1])
            action="remove"
    else:
        if rand <=0.5:
            R.add_edge(nodes[node1],nodes[node2])
            search_dag(R,node1,node2)
            action="right"
        else:
            R.add_edge(nodes[node2],nodes[node1])
            search_dag(R,node1,node2)
            action="left"
    ns=[nodes[node2],nodes[node1],action]
    return R,ns


def Best_Neighbor(agents,ind1):
    bestneighbor = agents[agents[ind1].neighbors[0]].score
    bestN = agents[ind1].neighbors[0]
    for i in range(1, 4):
        if bestneighbor > agents[agents[ind1].neighbors[i]].score:
            bestneighbor = agents[agents[ind1].neighbors[i]].score
            bestN = agents[ind1].neighbors[i]
    return bestN

def update_tabu(tabu,mov,tabu_length):
    if len(tabu)>=tabu_length:
        tabu.pop()
        tabu.insert(0,mov)
    else:
        tabu.insert(0,mov)

def find_best_neighbor(solution,solution_score,tabu,n,nodes,rdados,aux_bic,goalscore):
    all_tabu=True
    while(all_tabu):
        all_tabu=False
        neighbors=[]
        neighbors_score=[]
        neighbors_act=[]
        for i in range(n):
            [ind,b]=Mutation(solution,nodes)
            neighbors.append(ind)
            neighbors_act.append(b)
            [scr,aux_bic]=BIC_r(ind,rdados,aux_bic)
            neighbors_score.append(abs(scr))
        index=np.argsort(neighbors_score)
        aux=neighbors
        neighbors=[]
        aux2=neighbors_act
        neighbors_act=[]
        aux3=neighbors_score
        neighbors_score=[]
        for i in index:
            neighbors.append(aux[i])
            neighbors_act.append(aux2[i])
            neighbors_score.append(aux3[i])
        find=True
        i=0
        while(find):
            find=False
            if neighbors_score[i] >=solution_score:
               for j in range(len(tabu)):
                   if neighbors_act[i] == tabu[j]:
                       find=True
            i+=1
        if i>len(neighbors_act[i]):
            all_tabu=True
    return neighbors[i],neighbors_score[i],neighbors_act[i],aux_bic             

def tabu_search(solution,t_max,num_neighborhood,rdados,nodes,aux_bic,goalscore):
    best_solution=solution
    [scr,aux_bic]=BIC_r(best_solution,rdados,aux_bic)
    score_best=abs(scr)
    solution_score=score_best
    t=0
    best_t=0
    tabu=[]
    tabu_length=100
    while (t-best_t)<=t_max:
        t+=1
        [s,score,mov,aux_bic]=find_best_neighbor(solution,solution_score,tabu,num_neighborhood,nodes,rdados,aux_bic,goalscore)
        solution=s
        solution_score=score

        if score<score_best:
            best_solution=solution
            best_t=t
            score_best=score
        update_tabu(tabu,mov,tabu_length)       
    return best_solution,score_best,aux_bic

   
def MAGA_BN(nodes,Lsize,av_max,rdados,Pm_min,Pm_max,Po,Pc_min,Pc_max,num_neighborhood,t_max,goalscore,sL, sPm, sGen):
    besttemp=[]
    mediaagensts=[]
    BestScore=999999999
    Best=nx.DiGraph()
    Bestpos=0
    num_bic=0
    [Agents,num_bic]=createAgents(Lsize, rdados,nodes,num_bic)
    tempo_av=0
    for i in range(len(Agents)):
        tempo_av+=Agents[i].tempo
    t=0
    while (num_bic < av_max) and BestScore >(goalscore+0.00000001):
        Pc=(Pc_min-t*(Pc_min-Pc_max)/tmax)
        Pm=(Pm_min-t*(Pm_min-Pm_max)/tmax)
        #print(" geracao: {} PC: {} Po: {}".format(t,Pc,Po))
        t+=1
        best_ind=0
        for a in range(len(Agents)):
            aux_best_scr=Agents[a].score
            aux_best=a
            if random.uniform(0, 1) < Pc:
                bestN=Best_Neighbor(Agents,a)
                if Agents[a].score>Agents[bestN].score:
                    [new_agent,new_agent_score,num_bic]=Crossover(Agents[a].variable,Agents[bestN].variable,nodes,num_bic,goalscore)                    
                    Agents[a].newvariables(new_agent,new_agent_score)
                    aux_best_scr=new_agent_score                   
            if aux_best_scr<Agents[best_ind].score:
                best_ind=aux_best
        total_m=round(len(Agents)*len(Agents[0].variable)*Pm)
        aux_m=0
        while(aux_m<=total_m):
            aux_rand= random.randint(0,len(Agents)-1)
            if aux_rand !=best_ind:
                [new_agent,ns]=Mutation(Agents[aux_rand].variable,nodes)
                [scr,num_bic]=BIC_r(new_agent,rdados,num_bic)
                if abs(scr) <= Agents[aux_rand].score:
                    Agents[aux_rand].newvariables(new_agent,abs(scr))
                    aux_m=aux_m+1
                elif random.uniform(0, 1) < Po:
                    Agents[aux_rand].newvariables(new_agent,abs(scr))
                    aux_m=aux_m+1
        for a in Agents:
            if BestScore > a.score:
                BestScore = a.score
                Best = a.variable
                Bestpos = a.pos    
        [Agents[Bestpos[0]].variable,Agents[Bestpos[0]].score,num_bic]=tabu_search(Best,t_max,num_neighborhood,rdados,nodes,num_bic,goalscore)
        Best = Agents[Bestpos[0]].variable
        BestScore = Agents[Bestpos[0]].score
        Bestpos = Agents[Bestpos[0]].pos
        auxscore = 0
        print(" geracao: {} PC: {} Po: {}".format(t,Pc,Po))
        auxscore = 0
        for a in Agents:
            auxscore = auxscore + a.score
        besttemp.append(BestScore)
        mediaagensts.append(auxscore / len(Agents))
        t=t+1
    with open('DADOS_MAGA_Mestrado22_valores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(besttemp)
        writer.writerow(mediaagensts)
    return [Best,BestScore,t,tempo_av,num_bic]

problem='insurance100k.csv'
with open(problem) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    aux = 0
    data =[]
    data2=[]
    data1=[]
    prim=0
    for row in csv_reader:
        data.append(row)
        if prim==0:
            nodes=row
            print(nodes)
            data1=[ [] for i in range(len(nodes))]
            prim=1
        for i in range(len(row)):
            data1[i].append(row[i])
        aux=aux+1
        if aux == 100001:
            break
    data={}
for i in range(len(data1)):
    data[data1[i][0]]=[data1[i][j] for j in range(1,len(data1[i]))]
data = pd.DataFrame(data)
print("Data: ")
print(data) #Dados Retirandos do arquivo

tmax=200
av_max=100000
Pc_min =0.95
Pm_min = 0.01
Pc_max =0.95
Pm_max = 0.01 
Po = 0.05
Lsize = 10 
t_max=10
num_neighborhood=25
rdados=robjects.r('''
                  read.csv(file = 'insurance100k.csv')
        ''')

goalscore=993435.6
reader = BIFReader('insurance.bif')
asia_model = reader.get_model() # lendo esse modelo
num_bic=0
[scr,num_bic]=BIC_r(asia_model,rdados,num_bic)
goalscore=abs(scr)
print(goalscore)
times=20
print("--------------------------------------------")
print("Pc min: {} pc max: {} Pm min: {} Pm max: {} execucao: {}".format(Pc_min,Pc_max,Pm_min,Pm_max,i))
with open('DADOS_MAGA_Mestrado22.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["tmax", "Pc", "Pm","Po", "Lsize", "t_max", "num_neighborhood"])
    writer.writerow([tmax,[Pc_min,Pc_max],[Pm_min,Pm_max],Po,Lsize,t_max,num_neighborhood])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","Geraçoes","número avaliação do BIC","SLF","TLF","f1score", "accuracy","precision","recall"])
with open('DADOS_MAGA_Mestrado22_srt.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["tmax", "Pc", "Pm","Po", "Lsize", "t_max", "num_neighborhood"])
    writer.writerow([tmax,[Pc_min,Pc_max],[Pm_min,Pm_max],Po,Lsize,t_max,num_neighborhood])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    
    writer.writerow(["String"])
with open('DADOS_MAGA_Mestrado22_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["tmax", "Pc", "Pm","Po", "Lsize", "t_max", "num_neighborhood"])
    writer.writerow([tmax,[Pc_min,Pc_max],[Pm_min,Pm_max],Po,Lsize,t_max,num_neighborhood])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["valores"])    
for i in range(times):
    gc.collect()
    bestscores=[]
    tempos=[]
    temp_sec=[]
    BIC_vezes_vetor=[]
    num_bics=[]
    tempos_bic=[]
    temptotal = time.time()
    [Best,BestScore,t,tempo_bic,num_bic]=MAGA_BN(nodes,Lsize,av_max,rdados,Pm_min,Pm_max,Po,Pc_min,Pc_max,num_neighborhood,t_max,goalscore)
    temptotalf = time.time()-temptotal
    bestscores.append(BestScore)
    num_bics.append(num_bic)
    temp_sec.append(temptotalf)
    tempos.append(t)
    tempos_bic.append(tempo_bic)
    auxn=0
    srt_best=srt_network(Best,rdados,auxn)
    [SLF,TLF]=hamming2(asia_model,Best,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,Best,nodes)
    with open('DADOS_MAGA_Mestrado22.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temptotalf,BestScore,t,num_bic,SLF,TLF,f1score, accuracy,precision,recall])
    with open('DADOS_MAGA_Mestrado22_srt.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([srt_best])
    f = open("DADOS_MAGA_Mestrado22.txt", "a")
    f.write("tempo : {} BestScore: {} geracao: {} \n".format(temptotalf,BestScore,t))
    f.close()
no_conv=0
for d in range(len(bestscores)):
    if bestscores[d]>goalscore:
        no_conv+=1        
with open('DADOS_MAGA_Mestrado22_relatório.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["tmax", "Pc", "Pm","Po", "Lsize", "t_max", "num_neighborhood"])
            writer.writerow([tmax,[Pc_min,Pc_max],[Pm_min,Pm_max],Po,Lsize,t_max,num_neighborhood])
            writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
            writer.writerow(["Tempo médio",sum(temp_sec)/times, "Tempo total",sum(temp_sec),"Maior Tempo",max(temp_sec),"Menor Tempo",min(temp_sec),"Mediana Tempo",statistics.median(temp_sec)])
            writer.writerow(["Bic médio",sum(bestscores)/times, "Maior Bic",max(bestscores),"Menor Bic",min(bestscores),"Mediana Bic",statistics.median(bestscores)])
            writer.writerow(["Média de Gerações",sum(tempos)/times, "Maior Geração",max(tempos),"Menor Geração",min(tempos),"Mediana Geração",statistics.median(tempos)])
            writer.writerow(["Média números Bic",sum(num_bics)/times, "Maior números Bic",max(num_bics),"Menor números Bic",min(num_bics),"Mediana números Bic",statistics.median(num_bics)])
            writer.writerow(["Não convergiu", no_conv, " de ", times])
