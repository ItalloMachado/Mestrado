# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:44:17 2019
@author: Itallo Guilherme Machado
"""

import numpy as np

#import matplotlib.pyplot as plt  # to plot
import random
import networkx as nx
import pandas as pd
import csv
import time
from copy import deepcopy as deep_copy
from pgmpy.readwrite import BIFReader
from copy import deepcopy
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
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

def Mutation(ind1,nodes,prob,aux_bic,rdados):
    R = nx.DiGraph()
    R=ind1.copy()
    action="NNN"
    for a in nodes:
        if a not in R.nodes():
            R.add_node(a)
    node1= random.randint(0,len(nodes)-1)
    node2= node1
    verf=True
    while(verf):
        
        while(node1 == node2):
            node2= random.randint(0,len(nodes)-1)
        rand = random.random()
        if ind1.has_edge(nodes[node1],nodes[node2]):
            if rand <=0.5:
                R.remove_edge(nodes[node1],nodes[node2])
                R.add_edge(nodes[node2],nodes[node1])
           
                action="right"
                if nx.is_directed_acyclic_graph(R):
                    verf=False
                else:
                    R.remove_edge(nodes[node2],nodes[node1])
                    R.add_edge(nodes[node1],nodes[node2])
            else:
                R.remove_edge(nodes[node1],nodes[node2])
                action="remove"
                if nx.is_directed_acyclic_graph(R):
                    verf=False
                else:
                    R.add_edge(nodes[node1],nodes[node2])
                    
        elif ind1.has_edge(nodes[node2],nodes[node1]):
            if rand <=0.5:
                R.remove_edge(nodes[node2],nodes[node1])
                R.add_edge(nodes[node1],nodes[node2])
     
                action="left"
                if nx.is_directed_acyclic_graph(R):
                    verf=False
                else:
                    R.remove_edge(nodes[node1],nodes[node2])
                    R.add_edge(nodes[node2],nodes[node1])
                
                
            else:
                R.remove_edge(nodes[node2],nodes[node1])
                action="remove"
                if nx.is_directed_acyclic_graph(R):
                    verf=False
                else:
                    R.add_edge(nodes[node2],nodes[node1])
        else:
            if rand <=0.5:
                R.add_edge(nodes[node1],nodes[node2])
         
                action="right"
                if nx.is_directed_acyclic_graph(R):
                    verf=False
                else:
                    R.remove_edge(nodes[node1],nodes[node2])
                    
            else:
                R.add_edge(nodes[node2],nodes[node1])

                action="left"
                if nx.is_directed_acyclic_graph(R):
                    verf=False
                else:
                    R.remove_edge(nodes[node2],nodes[node1])

    ns=[nodes[node2],nodes[node1],action]
    [score,aux_bic]=BIC_r(R,rdados,aux_bic)
    score=abs(score)
    return R,score,aux_bic,ns


def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        return p
def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))
    
def  neighbor_search(bee,nodes,aux_bic,rdados):
    bee_aux=[deepcopy(bee),deepcopy(bee),deepcopy(bee),deepcopy(bee)] # 1 add 2 remove 3 reverse 4 move
    old_bee=deepcopy(bee)
    add=True
    cont=0
    cont_max=100
    while(add and cont<cont_max):
        cont+=1
        node1= random.randint(0,len(nodes)-1)
        node2= node1
        while(node1 == node2):
            node1= random.randint(0,len(nodes)-1)
            node2= random.randint(0,len(nodes)-1)
        if not bee_aux[0][0].has_edge(nodes[node1],nodes[node2]):
            bee_aux[0][0].add_edge(nodes[node1],nodes[node2])
            if nx.is_directed_acyclic_graph(bee_aux[0][0]):
                add=False
            else:
                 bee_aux[0][0].remove_edge(nodes[node1],nodes[node2])
    remove=True
    while(remove):
        node1= random.randint(0,len(nodes)-1)
        node2= node1
        while(node1 == node2):
            node2= random.randint(0,len(nodes)-1)
        if bee_aux[1][0].has_edge(nodes[node1],nodes[node2]):
            bee_aux[1][0].remove_edge(nodes[node1],nodes[node2])
            remove=False
    reverse=True
    while(reverse):
        node1= random.randint(0,len(nodes)-1)
        node2= node1
        while(node1 == node2):
            node2= random.randint(0,len(nodes)-1)
        if bee_aux[2][0].has_edge(nodes[node1],nodes[node2]):
            bee_aux[2][0].remove_edge(nodes[node1],nodes[node2])
            bee_aux[2][0].add_edge(nodes[node2],nodes[node1])
            if nx.is_directed_acyclic_graph(bee_aux[2][0]):
                reverse=False
            else:
                bee_aux[2][0].remove_edge(nodes[node2],nodes[node1])
                bee_aux[2][0].add_edge(nodes[node1],nodes[node2])
    cont=0
    new_bee=old_bee[0]
    new_score=old_bee[1]
    for i in range(3):
        [cost,aux_bic]=BIC_r(bee_aux[i][0],rdados,aux_bic) 
        cost=abs(cost)
        if cost<new_score:
            new_bee=bee_aux[i][0]
            new_score=cost
    return new_bee,new_score,aux_bic


def greedy_search(state, cost,step, num_bic,nodes,rdados,besttemp):
    valid=True
    print(cost)
    while valid:
        step+=1
        valid=False
        [new_state,new_cost,num_bic]=neighbor_search([state,cost],nodes,num_bic,rdados)
        print(new_cost)
        if cost>new_cost:
            state=deepcopy(new_state)
            cost=new_cost
            besttemp.append(cost)
            valid=True    
    return state,cost,num_bic,step,besttemp

def annealing(maxsteps,rdados,goalscore,av_max):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    prob=0.5
    aux_bic=0
    G=nx.fast_gnp_random_graph(len(nodes),0.1,directed=True)
    state = nx.DiGraph([(u,v,{'weight':1}) for (u,v) in G.edges() if u<v])
    random.shuffle(nodes)
    mapping={}
    random.shuffle(nodes)
    for k in range(len(nodes)):
        mapping.update({k:nodes[k]})
    state=nx.relabel_nodes(state, mapping)
    for a in nodes:
        if a not in state.nodes():
            state.add_node(a)
    [cost,aux_bic]=BIC_r(state,rdados,aux_bic)
    cost=abs(cost)
    states, costs = [state], [cost]
    step=0
    T=0.7
    besttemp=[]
    besttemp.append(state)
    while  (aux_bic < av_max) and cost >(goalscore+0.00000001):
        step+=1
        #print(step)
        [new_state,new_cost,aux_bic,ns] = Mutation(deep_copy(state),nodes,prob,aux_bic,rdados)
        if acceptance_probability(cost, new_cost, T) > random.random():
            state1= new_state.copy()
            cost=deep_copy(new_cost)
            states.append(state1)
            costs.append(cost)
            state=deep_copy(state1)
            besttemp.append(state)
    [cost,aux_bic]=BIC_r(state,rdados,aux_bic)
    cost=abs(cost)
    T=random.random()*T
    return state, cost,step, aux_bic,states, costs,besttemp


problem='child2k5.csv'
with open(problem) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    aux = 0
    data =[]
    data1=[]
    prim=0
    for row in csv_reader:
        data.append(row)
        if prim==0:
            nodes=row        
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

rdados=robjects.r('''
                  read.csv(file = 'child2k5.csv')
        ''')

reader = BIFReader('child.bif') # melhor rede do asia, como esta no bnlearn.com
asia_model = reader.get_model() # lendo esse modelo
print("Score BIC")
num_bic=0
[scr,num_bic]=BIC_r(asia_model,rdados,num_bic)
goalscore=abs(scr)
maxsteps=10000
times=20
num_bic=0
step=0
prob=0.5
aux_bic=0
av_max=60000

print("--------------------------------------------")
with open('SAGA_novo_mestrado3.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["maxsteps",maxsteps,"Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","Geraçoes","número avaliação do BIC","SLF","TLF","f1score", "accuracy","precision","recall"])

with open('SAGA_novo_mestrado3_srt.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["maxsteps",maxsteps,"Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","Geraçoes","número avaliação do BIC","SLF","TLF","f1score", "accuracy","precision","recall"])


with open('SAGA_novo_mestrado3_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["maxsteps",maxsteps,"Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
with open('SA2_mestrado3.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["maxsteps",maxsteps,"Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","Geraçoes","número avaliação do BIC","SLF","TLF","f1score", "accuracy","precision","recall"])
with open('SA2_mestrado3_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["maxsteps",maxsteps,"Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    
for i in range(times):
    gc.collect()
    bestscores=[]
    tempos=[]
    temp_sec=[]
    BIC_vezes_vetor=[]
    num_bics=[]
    tempos_bic=[]
    temptotal = time.time()
    [state, cost,step, num_bic,states, costs,besttemp] =annealing(maxsteps,rdados,goalscore,av_max)
    temptotalf = time.time()-temptotal
    [SLF,TLF]=hamming2(asia_model,state,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,state,nodes)
    print("SA: {}".format(cost))
    with open('SA.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temptotalf,cost,step,num_bic,SLF,TLF,f1score, accuracy,precision,recall])       
    f = open("SA2_mestrado3.txt", "a")
    f.write("tempo : {} BestScore: {} geracao: {} \n".format(temptotalf,cost,step))
    f.close()
    temptotal = time.time()
    [state,cost,num_bic,step,costs]=greedy_search(state, cost,step, num_bic,nodes,rdados,costs)
    temptotal = time.time()-temptotal
    temptotalf+=temptotal
    print("SAGA: {}".format(cost))
    print(costs)
    bestscores.append(cost)
    num_bics.append(num_bic)
    temp_sec.append(temptotalf)
    tempos.append(step)
    [SLF,TLF]=hamming2(asia_model,state,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,state,nodes)
    auxn=0
    srt_best=srt_network(state,rdados,auxn)
    with open('SAGA_novo_mestrado3.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temptotalf,cost,step,num_bic,SLF,TLF,f1score, accuracy,precision,recall])        
    with open('SAGA_novo_mestrado3_valores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([costs])
    with open('SAGA_novo_mestrado3_srt.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([srt_best])
    f = open("SAGA_novo_mestrado3.txt", "a")
    f.write("tempo : {} BestScore: {} geracao: {} \n".format(temptotalf,cost,step))
    f.close()
