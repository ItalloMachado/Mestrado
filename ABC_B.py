# -*- coding: utf-8 -*-
"""
Created on Tue Sat 13 19:33:08 2020

@author: Itallo Guilherme Machado
"""

import random
#import matplotlib.pyplot as plt
import numpy as np
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
    """Função que calcula o SLF e o TLF
    Entrada: target=Grafo desejado
             ind1 = Grafo que será comparado
             nodes= vetor com os nomes dos nós do problema
    Saída: valores das métricas SLF e TLF"""
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

def search_dag(G,edge_a,edge_b):
    """Operador de reparo: Verfica se o grafo possui ciclo, se tiver um ciclo 
    ele retira uma aresta do circulo que não seja a ultima que foi adicionada.
    Entrada: G=Grafo 
             edge_a e edge_b= nós da ultima aresta adicionada, sendo a->b.
    Saída: Gráfo com DAG
             """
    
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


def roleta(prob):
    """ Função que escolhe um indivíduo através de uma roleta.
    entrada: prob= vetor de probabilidades de  cada indivíduo
    Saída: escolhido= indivíduo escolhido."""   
    soma_prob=sum(prob)
    prob=[prob[i]/soma_prob for i in range(len(prob))]
    rand_num=random.random()
    aux_num=0
    i=0
    while aux_num<rand_num:
        aux_num+=prob[i]
        i+=1
    escolhido=i-1    
    return escolhido


def Probabilistic_transition_rule_v2_new(pheromones_matrix,bee,nodes,q0,alpha,beta,score_matrix,score_old,aux_bic):
   
    bee_aux=[deepcopy(bee),deepcopy(bee)]
    eta_soma=0
    eta_aux=[]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if not bee_aux[0][0].has_edge(nodes[i],nodes[j]) or not bee_aux[0][0].has_edge(nodes[j],nodes[i]): 
                bee_aux[0][0].add_edge(nodes[i],nodes[j])
                if nx.is_directed_acyclic_graph(bee_aux[0][0]):
                    cost=score_matrix[i][j]
                    aux=((cost-score_old)**beta)*pheromones_matrix[i][j]**alpha
                    eta_soma+=aux
                    eta_aux.append([nodes[i],nodes[j],aux])
                else:
                    eta_aux.append([nodes[i],nodes[j],-9999999])
                bee_aux[0][0].remove_edge(nodes[i],nodes[j])
    return eta_aux,eta_soma


def Probabilistic_transition_rule_v2_aux(eta_aux,eta_soma,aux_bic):
    q=random.random()
    if q<=q0:
        max_eta=eta_aux[0]
        for i in range(len(eta_aux)):
            if max_eta[2]<eta_aux[i][2]:
                max_eta=eta_aux[i]
        return max_eta,aux_bic
    else:
        prob=[]
        for i in range(len(eta_aux)):
            if eta_aux[i][2] == -9999999:
                prob.append(0)
            else:
                prob.append(eta_aux[i][2]/eta_soma)                
        esc=roleta(prob)
        return eta_aux[esc],aux_bic         
    return
   

def neighbor_KG_search_v2(bee,score_matriz,nodes,rdados):
    aux_bic=1
    bee_aux=[deepcopy(bee),deepcopy(bee)]
    new_bee_add=[[0,0],999999]
    new_bee_remove=[[0,0],0]
    for i in nodes:
        for j in nodes:         
            if not bee_aux[0][0].has_edge(i,j):                
                bee_aux[0][0].add_edge(i,j)
                if nx.is_directed_acyclic_graph(bee_aux[0][0]):
                    ix_i=nodes.index(i)
                    ix_j=nodes.index(j)
                    cost=score_matriz[ix_i][ix_j] 
                    cost=abs(cost)
                    if new_bee_add[1]>cost:
                        new_bee_add[0]=[i,j]
                        new_bee_add[1]=cost
                bee_aux[0][0].remove_edge(i,j)
            if bee_aux[0][0].has_edge(i,j):
                ix_i=nodes.index(i)
                ix_j=nodes.index(j)
                cost=score_matriz[ix_i][ix_j] 
                cost=abs(cost)
                if new_bee_remove[1]<cost:
                    new_bee_remove[0]=[i,j]
                    new_bee_remove[1]=cost
    bee_add=bee_aux[0][0].add_edge(new_bee_add[0][0],new_bee_add[0][1])
    [cost_add,aux_bic]=BIC_r(bee_aux[0][0],rdados,aux_bic)
    cost_add=abs(cost_add)
    bee_aux[0][0].remove_edge(new_bee_add[0][0],new_bee_add[0][1])
    bee_add=bee_aux[0][0].add_edge(new_bee_add[0][1],new_bee_add[0][0])
    if nx.is_directed_acyclic_graph(bee_aux[0][0]):
        [cost_add2,aux_bic]=BIC_r(bee_aux[0][0],rdados,aux_bic)
        cost_add2=abs(cost_add2)
        bee_aux[0][0].remove_edge(new_bee_add[0][1],new_bee_add[0][0])
        if cost_add>cost_add2:
            cost_add=cost_add2
            aux=new_bee_add[0][1]
            new_bee_add[0][1]=new_bee_add[0][0]
            new_bee_add[0][0]=aux        
    else:
        bee_add=bee_aux[0][0].remove_edge(new_bee_add[0][1],new_bee_add[0][0])  
    bee_remove=bee_aux[0][0].remove_edge(new_bee_remove[0][0],new_bee_remove[0][1])
    [cost_remove,aux_bic]=BIC_r(bee_aux[0][0],rdados,aux_bic)
    cost_remove=abs(cost_remove)
    if cost_add<=bee[1] and cost_add<=cost_remove:
        return bee_add,cost_add
    elif cost_remove<=bee[1] and cost_add>cost_remove:
        return bee_remove,cost_remove


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
            search_dag(bee_aux[0][0],nodes[node1],nodes[node2])
            add=False
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
            search_dag(bee_aux[2][0],nodes[node2],nodes[node1])        
            reverse=False
    move=True
    cont=0
    while(move and cont<300):
        cont+=1
        node1= random.randint(0,len(nodes)-1)
        node2= node1
        while(node1 == node2):
            node2= random.randint(0,len(nodes)-1)
        parents_node1=list(bee_aux[3][0].predecessors(nodes[node1]))
        parents_node2=list(bee_aux[3][0].predecessors(nodes[node2]))
        if parents_node1 and parents_node2:  
            parent1= random.randint(0,len(parents_node1)-1)
            parent2= random.randint(0,len(parents_node2)-1)
            if parents_node1 != parents_node2:
                while parents_node1[parent1] == parents_node2[parent2]:
                    parent1= random.randint(0,len(parents_node1)-1)
                    parent2= random.randint(0,len(parents_node2)-1)
                if parents_node1[parent1] not in parents_node2 and parents_node2[parent2] not in parents_node1:
                    if parents_node1[parent1] != nodes[node2] and parents_node2[parent2] != nodes[node1]:
                        aux_nodes=deepcopy(nodes)
                        index_parant1=nodes.index(parents_node1[parent1])
                        index_parant2=nodes.index(parents_node2[parent2])
                        aux=deepcopy(aux_nodes[index_parant2])
                        aux_nodes[index_parant2]=aux_nodes[index_parant1]
                        aux_nodes[index_parant1]=aux
                        mapping={}    
                        for k in range(len(aux_nodes)):
                            mapping.update({nodes[k]:aux_nodes[k]})
                        bee_aux[3][0]=nx.relabel_nodes(bee_aux[3][0], mapping)
                        [cost,aux_bic]=BIC_r(bee_aux[3][0],rdados,aux_bic)                    
                        cost=abs(cost)
                        move=False
    new_bee=old_bee[0]
    new_score=old_bee[1]
    for i in range(4):
        [cost,aux_bic]=BIC_r(bee_aux[i][0],rdados,aux_bic) 
        cost=abs(cost)
        if cost<new_score:
            new_bee=bee_aux[i][0]
            new_score=cost
    return new_bee,new_score,aux_bic
         
        
def ABC_B(K,N_max,q0,qd,alpha,beta,p,limit,rdados,nodes,av_max):
    t=0
    aux_bic=0   
    bees=[]
    net=nx.DiGraph()
    for a in nodes:
        net.add_node(a) 
    pheromones_matrix=[]
    for i in nodes:
        pheromones_matrix_aux=[]
        for j in nodes:
            if i==j:
                pheromones_matrix_aux.append(0)                
            else:
                net.add_edge(i,j)
                [cost,aux_bic]=BIC_r(net,rdados,aux_bic)                
                cost=abs(cost)
                pheromones_matrix_aux.append(1/(len(nodes)*cost))
                net.remove_edge(i,j)
        pheromones_matrix.append(pheromones_matrix_aux)    
    for i in range(K):
        G=nx.fast_gnp_random_graph(len(nodes),0.3,directed=True)
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
        bees.append([state,cost])
    t=0
    best_solution=bees[0]
    state = nx.DiGraph()
    for a in nodes:
        state.add_node(a)
    [cost_inicial,aux_bic]=BIC_r(state,rdados,aux_bic)
    cost_inicial=1
    cost_inicial=abs(cost_inicial)
    score=99999999999  
    score_matriz=[]    
    score_matriz=[]
    for i in nodes:
        score_matriz_aux=[]
        for j in nodes:
            if i!=j:
                state.add_edge(i,j)
                [cost,aux_bic]=BIC_r(state,rdados,aux_bic)                
                cost=abs(cost)
                score_matriz_aux.append(cost)
                state.remove_edge(i,j)
                if score>cost:
                    score=cost
            else:
                score_matriz_aux.append(0)
        score_matriz.append(score_matriz_aux)
    score_matriz=np.array(score_matriz)
    bees=np.array(bees)
    pheromones_matrix=np.array(pheromones_matrix)
    besttemp=[]
    while (aux_bic < av_max) and best_solution[1]>(goalscore+0.00000001):   
        #Neighbor search phase of employed bees
        t+=1
        #print(t)
        old_bees=deepcopy(bees)
        for i in range(K):
            [new_bee,new_score,aux_bic]=neighbor_search(bees[i],nodes,aux_bic,rdados)
            if bees[i][1]>new_score:
                bees[i][0]=new_bee
                bees[i][1]=new_score
        #Neighbor search phase of onlookers
        soma_score=0
        for i in range(K):
            soma_score+=bees[i][1]
        prob_bee=[bees[i][1]/soma_score for i in range(K)]
        
        for i in range(K):
            chose=roleta(prob_bee)
            q=random.random()
            if q<qd:
                [new_bee,new_score]=neighbor_KG_search_v2(bees[chose],score_matriz,nodes,rdados)
            else:

                [new_bee,new_score,aux_bic]=neighbor_search(bees[chose],nodes,aux_bic,rdados)
            if bees[i][1]>new_score:
                bees[i][0]=new_bee
                bees[i][1]=new_score                
        #Exploring new solutions by scouts
        Ck=0
        for i in range(K):
            if bees[i][1] == old_bees[i][1]:
                Ck+=1
            else:
                Ck=0
            if Ck==limit:                
                bees[i]=bees[i]                
                net=nx.DiGraph()
                for a in nodes:
                    net.add_node(a)
                [cost,aux_bic]=BIC_r(net,rdados,aux_bic)
                cost=abs(cost)
                new_bee=[net,cost]
                old_bee=[new_bee[0],cost+1]
                key=True
                pheromones_matrix_2=pheromones_matrix[:]
                [eta_aux,eta_soma]=Probabilistic_transition_rule_v2_new(pheromones_matrix_2,new_bee,nodes,q0,alpha,beta,score_matriz,cost_inicial,aux_bic)
                while old_bee[1]>new_bee[1] or key:
                    key= False        
                    old_bee=new_bee[:]
                    [edge_add,aux_bic]=Probabilistic_transition_rule_v2_aux(eta_aux,eta_soma,aux_bic)
                    if new_bee[0].has_edge(edge_add[1],edge_add[0]):
                        index_i=nodes.index(edge_add[0])
                        index_j=nodes.index(edge_add[1])                
                        pheromones_matrix_2[index_i][index_j]=0
                        key=True
                    else:        
                        new_bee[0].add_edge(edge_add[0],edge_add[1])
                        search_dag(new_bee[0],edge_add[0],edge_add[1])                            
                    [cost,aux_bic]=BIC_r(new_bee[0],rdados,aux_bic)
                    new_bee[1]=abs(cost)
                bees[i]=deepcopy(old_bee)                                                
                Ck=0
        #Memorize the best solution G found so far
        for i in range(K):
            if best_solution[1]>bees[i][1]:
                best_solution=deepcopy(bees[i])       
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if best_solution[0].has_edge(nodes[i],nodes[j]):
                    delta_pheromone=1/best_solution[1]
                else:
                    delta_pheromone=pheromones_matrix[i][j]                    
                pheromones_matrix[i][j]=(1-p)*pheromones_matrix[i][j]+p*delta_pheromone
        besttemp.append(best_solution[1])    
    return best_solution[0],best_solution[1],aux_bic,t,besttemp
   
            

            
        
        
problem='insurance100k.csv'       
        
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
            if aux == 200001:
                break
        data={}
for i in range(len(data1)):
    data[data1[i][0]]=[data1[i][j] for j in range(1,len(data1[i]))]
data = pd.DataFrame(data)
print("Data: ")
print(data) #Dados Retirandos do arquivo\
rdados=robjects.r('''
                  read.csv(file = 'insurance100k.csv') 
        ''')
K=40
N_max=200
q0=0.8
qd=0
alpha=1
beta=2
p=1
limit=3
av_max=180000


reader = BIFReader('insurance.bif') # melhor rede do asia, como esta no bnlearn.com
asia_model = reader.get_model() # lendo esse modelo
print("Goal Score BIC")

[cost,aux_bic]=BIC_r(asia_model,rdados,1)
goalscore=abs(cost)
print(goalscore)

times=1

print("--------------------------------------------")
with open('DADOS_R_ABC_mestrado31.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["tempo", "BestScore","Geraçoes","número avaliação do BIC","SLF","TLF","f1score", "accuracy","precision","recall"])
with open('DADOS_R_ABC_mestrado31_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
with open('DADOS_R_ABC_mestrado31_srt.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
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
    [best_solution,score,num_bic,t,besttemp]=ABC_B(K,N_max,q0,qd,alpha,beta,p,limit,rdados,nodes,av_max)
    temptotalf = time.time()-temptotal
    temp_sec.append(temptotalf)
    [SLF,TLF]=hamming2(asia_model,best_solution,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,best_solution,nodes)
    auxn=0
    srt_best=srt_network(best_solution,rdados,auxn)
    with open('DADOS_R_ABC_mestrado31.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temptotalf,score,t,num_bic,SLF,TLF,f1score, accuracy,precision,recall])
    with open('DADOS_R_ABC_mestrado31_srt.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([srt_best])
    with open('DADOS_R_ABC_mestrado31_valores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([besttemp])
