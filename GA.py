# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:08:13 2019

@author: Itallo
"""

import random
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

class agent:
    def __init__(self, variable, rdados,scr=0):
        aux2=0
        self.variable = variable
        temp = time.time()
        if scr == 0:
            [scr,aux2]=BIC_r(variable,rdados,aux2)
            self.score = abs(scr)
        else:
            self.score=scr
        temp = time.time()-temp
        self.tempo=temp
    def newvariables(self, val, scr):
        self.variable = val
        self.score = scr

def createAgents(Lsize, rdados,nodes,aux_bic):
    agents = []
    cont=0
    for i in range(Lsize):
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
        agents.append(agent(x, rdados))
        aux_bic+=1
        cont+=1
    return agents,aux_bic


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


def Crossover(ind1,ind2,nodes,aux_bic):
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


def torneio(Agents,pop_size):
    k=0.75
    individuo1=round((pop_size-1) * random.random())
    individuo2=round((pop_size-1) * random.random())
    r = random.random()
    if r<=k:
        if Agents[individuo1].score<=Agents[individuo2].score:
            return individuo1
        else:
            return individuo2
    else:
        if Agents[individuo1].score>=Agents[individuo2].score:
            return individuo1
        else:
            return individuo2


def Elitismo(all_agents,pop_size):
    prox_ind=[]
    fit=[]
    melhorfit=all_agents[0].score
    melhorind=all_agents[0].variable
    for j in range(len(all_agents)):
        fit.append(all_agents[j].score)
        if all_agents[j].score<melhorfit:
            melhorfit=all_agents[j].score
            melhorind=all_agents[j].variable
            
    indice=[i[0] for i in sorted(enumerate(fit), key=lambda x:x[1])] # sort
    i=0
    while len(prox_ind)<pop_size:
        prox_ind.append(all_agents[indice[i]])
        i=i+1
    return prox_ind,melhorfit,melhorind

inicio = time.time()
save_costs_GA=[]
save_costs_GA_H=[]

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


min_valor=0
max_valor=2
pop_size=100
p_mutacao=0.01
p_cruzamento=0.7
gen_max=200
times=20
av_max=40000  
            
reader = BIFReader('child.bif') # melhor rede do asia, como esta no bnlearn.com
asia_model = reader.get_model() # lendo esse modelo
print("Goal Score BIC")
num_bic=0
rdados=robjects.r('''
                  read.csv(file = 'child500.csv')
        ''')
[scr,num_bic]=BIC_r(asia_model,rdados,num_bic)
goalscore=abs(scr)
print(goalscore)
with open('DADOS_GA_mestrado2.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["gen_max", "p_cruzamento", "p_mutacao","pop_size"])
    writer.writerow([gen_max,p_cruzamento,p_mutacao,pop_size,pop_size])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore])            
    writer.writerow(["tempo", "BestScore","Geraçoes","número avaliação do BIC","SLF","TLF","f1score", "accuracy","precision","recall"])
with open('DADOS_GA_mestrado2_valores.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["gen_max", "p_cruzamento", "p_mutacao","pop_size"])
    writer.writerow([gen_max,p_cruzamento,p_mutacao,pop_size,pop_size])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore])            
with open('DADOS_GA_Mestrado2_srt.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["gen_max", "p_cruzamento", "p_mutacao","pop_size"])
    writer.writerow([gen_max,p_cruzamento,p_mutacao,pop_size,pop_size])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore])     
for r in range(times):
    gc.collect()
    temptotalf=time.time()
    fitness=[]
    temp_sec=[]
    bestscores=[]
    tempos=[]
    nao_dag=[]
    num_bics=[]
    num_bic=0    
    ind_size=round((len(nodes)*len(nodes)-len(nodes))/2)    
    besttemp=[]
    mediaagensts=[]
    gen=0
    melhor_fit=[]
    #populacao inicial
    [Agents,num_bic]=createAgents(pop_size, rdados,nodes,num_bic)
    melhor_fitness=Agents[0].score
    while num_bic<av_max and melhor_fitness >(goalscore+0.00000001):
        auxscore=0
        print(gen)
        new_agents = []
        while len(new_agents)<pop_size:
            ind_sel1=torneio(Agents,pop_size)
            ind_sel2=torneio(Agents,pop_size)
            if random.uniform(0, 1) < p_cruzamento:
                [new_agent,new_agent_score,num_bic]=Crossover(Agents[ind_sel1].variable,Agents[ind_sel2].variable,nodes,num_bic)
                #temp2 = time.time()
                new_agents.append(agent(new_agent,rdados,new_agent_score))           
        total_m=round(len(new_agents)*len(new_agents[0].variable)*p_mutacao)
        aux_m=0
        while(aux_m<=total_m):
            aux_rand= random.randint(0,len(new_agents)-1)
            [new_agent,ns]=Mutation(new_agents[aux_rand].variable,nodes)
            [scr,num_bic]=BIC_r(new_agent,rdados,num_bic)
            new_agents[aux_rand].newvariables(new_agent,abs(scr))
            aux_m=aux_m+1
        all_agents=Agents+new_agents
        Agents=[]
        [Agents,melhor_fitness,melhorind]=Elitismo(all_agents,pop_size)
        melhor_fit.append(melhor_fitness)
        for a in Agents:
            auxscore = auxscore + a.score
        besttemp.append(melhor_fitness)
        mediaagensts.append(auxscore / len(Agents))       
        gen=gen+1
    temptotalf=time.time()-temptotalf
    temp_sec.append(temptotalf)
    bestscores.append(melhor_fitness)
    tempos.append(gen)
    num_bics.append(num_bic)
    [SLF,TLF]=hamming2(asia_model,melhorind,nodes)
    [f1score, accuracy,precision,recall]=F1Score(asia_model,melhorind,nodes)
    auxn=0
    srt_best=srt_network(melhorind,rdados,auxn,goalscore)
    with open('DADOS_GA_mestrado2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([temptotalf,melhor_fitness,gen,num_bic,SLF,TLF,f1score, accuracy,precision,recall])
            
    print("melhor score:{}".format(melhor_fitness))
    save_costs_GA.append(melhor_fitness)
    with open('DADOS_GA_Mestrado2_srt.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([srt_best])
    with open('DADOS_GA_mestrado2_valores.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(besttemp)
        writer.writerow(mediaagensts)
        
no_conv=0
for d in range(len(bestscores)):
    if bestscores[d]>goalscore:
        no_conv+=1
with open('DADOS_GA_mestrado2_relatório.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["gen_max", "p_cruzamento", "p_mutacao","pop_size"])
    writer.writerow([gen_max,p_cruzamento,p_mutacao,pop_size,pop_size])
    writer.writerow(["Problema: ",problem,"Score: ",goalscore,"Vezes: ",times])
    writer.writerow(["Tempo médio",sum(temp_sec)/times, "Tempo total",sum(temp_sec),"Maior Tempo",max(temp_sec),"Menor Tempo",min(temp_sec),"Mediana Tempo",statistics.median(temp_sec)])
    writer.writerow(["Bic médio",sum(bestscores)/times, "Maior Bic",max(bestscores),"Menor Bic",min(bestscores),"Mediana Bic",statistics.median(bestscores)])
    writer.writerow(["Média de Gerações",sum(tempos)/times, "Maior Geração",max(tempos),"Menor Geração",min(tempos),"Mediana Geração",statistics.median(tempos)])
    writer.writerow(["Média números Bic",sum(num_bics)/times, "Maior números Bic",max(num_bics),"Menor números Bic",min(num_bics),"Mediana números Bic",statistics.median(num_bics)])
    writer.writerow(["Não convergiu", no_conv, " de ", times])

