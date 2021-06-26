# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:05:00 2020

@author: Itallo
"""

#import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import csv
from pgmpy.readwrite import BIFReader
import time
from copy import deepcopy
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import gc

Bnl = importr('bnlearn')

rscore=robjects.r['score']
fn_score='bde'


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

def FES(state,nodes,rdados,aux_bic,costs,goalscore):
    again=True
    [cost,aux_bic]=BIC_r(state,rdados,aux_bic)
    while(again):
        aux_state=deepcopy(state)
        new_state=deepcopy(state)
        cost=abs(cost)
        best_cost=cost
        no_add=True
        if best_cost<(goalscore+0.00000001):
            break;
        for i in nodes:
            for j in nodes:
                if i !=j:
                    if not aux_state.has_edge(i,j):                
                        aux_state.add_edge(i,j)
                        if nx.is_directed_acyclic_graph(aux_state):
                            [cost,aux_bic]=BIC_r(aux_state,rdados,aux_bic) 
                            cost=abs(cost)
                            if best_cost>cost:
                                no_add=False
                                new_state=deepcopy(aux_state)
                                best_cost=cost
                                costs.append(best_cost)
                            else:
                                costs.append(best_cost)
                        aux_state.remove_edge(i,j)
        state=deepcopy(new_state)
        cost=best_cost
        if no_add:
            again=False
    return new_state,best_cost,aux_bic,costs

def BES(state,nodes,rdados,aux_bic,costs,goalscore):
    again=True
    [cost,aux_bic]=BIC_r(state,rdados,aux_bic)
    while(again):
        aux_state=deepcopy(state)
        new_state=deepcopy(state)
        cost=abs(cost)
        best_cost=cost
        if best_cost<(goalscore+0.00000001):
            break;            
        no_add=True
        for i in nodes:
            for j in nodes:
                if i !=j:
                    if aux_state.has_edge(i,j):                
                        aux_state.remove_edge(i,j)
                        if nx.is_directed_acyclic_graph(aux_state):
                            [cost,aux_bic]=BIC_r(aux_state,rdados,aux_bic) 
                            cost=abs(cost)
                            #print(cost)
                            if best_cost>cost:
                                no_add=False
                                new_state=deepcopy(aux_state)
                                best_cost=cost
                                costs.append(best_cost)
                            else:
                                costs.append(best_cost)
                        aux_state.add_edge(i,j)
        state=deepcopy(new_state)
        cost=best_cost
        if no_add:
            again=False
    return new_state,best_cost,aux_bic,costs

def GES(nodes,rdados,goalscore):
    aux_bic=0
    state = nx.DiGraph()
    for a in nodes:
        state.add_node(a)
    costs=[]
    [state,cost,aux_bic,costs]=FES(state,nodes,rdados,aux_bic,costs,goalscore)
    [state,cost,aux_bic,costs]=BES(state,nodes,rdados,aux_bic,costs,goalscore)
    return state,cost,aux_bic,costs

problem='asia400.csv'
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
                print(nodes)
                data1=[ [] for i in range(len(nodes))]
                prim=1
            for i in range(len(row)):
                data1[i].append(row[i])
            aux=aux+1
            if aux == 2000001:
                break
        data={}
for i in range(len(data1)):
    data[data1[i][0]]=[data1[i][j] for j in range(1,len(data1[i]))]
data = pd.DataFrame(data)
print("Data: ")
print(data) #Dados Retirandos do arquivo\
rdados=robjects.r('''
                  read.csv(file = 'asia400.csv')
        ''')


reader = BIFReader('asia.bif') # melhor rede do asia, como esta no bnlearn.com
asia_model = reader.get_model() # lendo esse modelo
print("Goal Score BIC")
[cost,aux_bic]=BIC_r(asia_model,rdados,1)
goalscore=abs(cost)
print(goalscore)

times=20
print("--------------------------------------------")
with open('GES_mestrado.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","número avaliação do BIC"])
with open('GES_mestrado_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","número avaliação do BIC"])
with open('GES_mestrado_srt1.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","número avaliação do BIC"])
for i in range(times):
    gc.collect()
    print(i)
    bestscores=[]
    tempos=[]
    temp_sec=[]
    BIC_vezes_vetor=[]
    num_bics=[]
    tempos_bic=[]
    temptotal = time.time()
    [state,cost,num_bic,costs]=GES(nodes,rdados,goalscore)
    temptotalf = time.time()-temptotal
    bestscores.append(cost)
    num_bics.append(num_bic)
    temp_sec.append(temptotalf)
    [SLF,TLF]=hamming2(asia_model,state,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,state,nodes)
    auxn=0
    srt_best=srt_network(state,rdados,auxn)
    with open('GES_mestrado.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temptotalf,cost,num_bic,SLF,TLF,f1score, accuracy,precision,recall])
    with open('GES_mestrado_valores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([costs])
    with open('GES_mestrado_srt1.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([srt_best])
