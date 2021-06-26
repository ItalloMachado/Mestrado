# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:57:03 2020

@author: Itallo Guilherme Machado
"""


import random
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


class particle:
    def __init__(self,network,variable, rdados):
        aux2=0
        self.matrix = variable
        temp = time.time()
        self.network = network
        [scr,aux2]=BIC_r(network,rdados,aux2)
        self.score = abs(scr)
        temp = time.time()-temp
        self.tempo=temp


def DFS(u,color,net): 
# GRAY :  This vertex is being processed (DFS 
#         for this vertex has started, but not 
#         ended (or this vertex is in function 
#         call stack) 
    u_idx=nodes.index(u)
    color[u_idx] = "GRAY"
    for v in list(net.neighbors(u)):
        v_idx=nodes.index(v)
        if color[v_idx] == "GRAY": 
            return True      
        if color[v_idx] == "WHITE" and DFS(v, color,net) == True: 
            return True      
    color[u_idx] = "BLACK"
    return False
  
def is_Cyclic(nodes,net):
    color = ["WHITE"] * len(nodes)
    for i in range(len(nodes)): 
        if color[i] == "WHITE": 
            if DFS(nodes[i], color,net) == True: 
                return True
    return False 
# Driver program to test above functions 
                               
# This program is contributed by Divyanshu Mehta  
def cov_matrix_network(matrix,nodes):
    n=len(nodes)
    G = nx.DiGraph()
    for i in range(n):
        aux=[]
        for j in range(n):
            aux.append(matrix[j+i*8])
            if matrix[j+i*n] == 1:
                G.add_edge(nodes[i],nodes[j])
    return G

def cov_network_matrix(G,nodes):
    n=len(nodes)
    matrix=[0]*n*n
    for i in range(n):
        for j in list(G.neighbors(nodes[i])):
             j_idx=nodes.index(j)
             matrix[j_idx+i*n]=1
    return matrix 


def createAgents(pop, rdados,nodes,aux_bic):
    particles = []
    cont=0
    for i in range(pop):
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
        particles.append(particle(x,cov_network_matrix(x,nodes),rdados))
        aux_bic+=1
        cont+=1
    return particles,aux_bic


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



def Mutation_MAGA(ind1,nodes):
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
    R_matrix=cov_network_matrix(R,nodes)
    return R_matrix,R,ns


def Crossover_MAGA(ind1,ind2,nodes,aux_bic):
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
        R_matrix=cov_network_matrix(f1,nodes)
        return R_matrix,f1,score_f1,aux_bic
    else:
        R_matrix=cov_network_matrix(f2,nodes)
        return R_matrix,f2,score_f2,aux_bic


def BNC_PSO(goalscore,data,nodes,pop,evaluations,w_start,w_end,c1_start,c1_end,c2_start,c2_end,av_max):
    aux_bic=0
    [particles,aux_bic]=createAgents(pop, data,nodes,aux_bic)
    best_particle=particles[0].score
    media_score=0
    matrix_best_particle=[]
    network_best_particle=[]
    for i in range(len(particles)):  
        if particles[i].score<best_particle:
            best_particle=particles[i].score
            matrix_best_particle=particles[i].matrix
            network_best_particle=particles[i].network
    ev=0
    bests=[]
    medias=[]
    while  (aux_bic < av_max) and best_particle >(goalscore+0.00000001):
        #print("ev: {}".format(ev))
        #print("aux_bic: {}".format(aux_bic))
        w=w_start-((w_start-w_end)/evaluations)*ev
        c1=c1_start-((c1_start-c1_end)/evaluations)*ev
        c2=c2_start-((c2_start-c2_end)/evaluations)*ev
        new_particles=[]
        for i in range(len(particles)):
            w_rand=random.random()
            if w_rand<w:
                [aux_matrix,aux_G,ns]=Mutation_MAGA(particles[i].network,nodes)                
            else:
                aux_matrix=deepcopy(particles[i].matrix)
                aux_G=deepcopy(particles[i].network)
            c1_rand=random.random()
            if c1_rand<c1:
                rand_index=random.randint(0,len(particles)-1)
                while rand_index == i:
                    rand_index=random.randint(0,len(particles)-1)
                    rand_G=particles[rand_index].network
                    [aux_matrix,aux_G,score_f1,aux_bic]=Crossover_MAGA(aux_G,rand_G,nodes,aux_bic)              
            c2_rand=random.random()
            if c2_rand<c2:
                if matrix_best_particle != aux_matrix:
                    [aux_matrix,aux_G,score_f1,aux_bic]=Crossover_MAGA(aux_G,network_best_particle,nodes,aux_bic)
            new_particles.append(particle(aux_G,aux_matrix,rdados))
        media_score=0
        for i in range(len(new_particles)):

            if new_particles[i].score<best_particle:
                best_particle=deepcopy(new_particles[i].score)
                matrix_best_particle=deepcopy(particles[i].matrix)
                network_best_particle=deepcopy(particles[i].network)
        for i in range(len(particles)):
           if new_particles[i].score<=particles[i].score:
               particles[i].score=deepcopy(new_particles[i].score)
               particles[i].matrix=deepcopy(new_particles[i].matrix)
               particles[i].network=deepcopy(new_particles[i].network)
    
        for i in range(len(particles)):
            media_score+=particles[i].score
        bests.append(best_particle)
        medias.append(media_score/pop)
        ev+=1
    return network_best_particle , best_particle,ev,aux_bic,bests,medias



problem='child500.csv'
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

rdados=robjects.r('''
              read.csv(file = 'child500.csv')
    ''')
pop=100
evaluations=200
w_start=0.95
w_end=0.4
c1_start=0.82
c1_end=0.5
c2_start=0.4
c2_end=0.83
av_max=12000


reader = BIFReader('child.bif') # melhor rede do asia, como esta no bnlearn.com
asia_model = reader.get_model() # lendo esse modelo
print("Goal Score BIC")
[cost,aux_bic]=BIC_r(asia_model,rdados,1)
goalscore=abs(cost)
print(goalscore)

times=1
w_starts=[0.95]
w_ends=[0.4]
print("--------------------------------------------")
print("pop: {} evaluations: {} execucao: {}".format(pop,evaluations,i))
with open('BNC_PSO_mestrado41.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["pop", "evaluations"])
    writer.writerow([pop,evaluations])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times,"w_start:",w_start,"w_end",w_end])
    writer.writerow(["tempo", "BestScore", "Geracao","número avaliação do BIC"])
with open('BNC_PSO_mestrado41_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times,"w_start:",w_start,"w_end",w_end])
with open('BNC_PSO_mestrado41_srt.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times,"w_start:",w_start,"w_end",w_end])    
for i in range(times):
    gc.collect()
    bestscores=[]
    tempos=[]
    temp_sec=[]
    BIC_vezes_vetor=[]
    num_bics=[]
    tempos_bic=[]
    temptotal = time.time()
    [network_best_particle , best_particle,t,aux_bic,bests,medias]=BNC_PSO(goalscore,rdados,nodes,pop,evaluations,w_start,w_end,c1_start,c1_end,c2_start,c2_end,av_max)
    temptotalf = time.time()-temptotal
    bestscores.append(best_particle)
    temp_sec.append(temptotalf)
    tempos.append(t)
    [SLF,TLF]=hamming2(asia_model,network_best_particle,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,network_best_particle,nodes)
    auxn=0
    srt_best=srt_network(network_best_particle,rdados,auxn)
    with open('BNC_PSO_mestrado41.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([temptotalf,best_particle,t,aux_bic,SLF,TLF,f1score, accuracy,precision,recall])
    with open('BNC_PSO_mestrado41_valores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([bests])
        writer.writerow([medias])
    with open('BNC_PSO_mestrado41_srt.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([srt_best])
    f = open("BNC_PSO_mestrado41.txt", "a")
    f.write("tempo : {} BestScore: {} geracao: {} \n".format(temptotalf,best_particle,t))
    f.close()
no_conv=0
for d in range(len(bestscores)):
    if bestscores[d]>goalscore:
        no_conv+=1
                
    