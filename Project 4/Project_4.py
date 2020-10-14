"""Scientific Computation Project 1
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy.special import expit #sigmoid function
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import ticker, cm


def data(fname='Inter_country_trans_data'):
    """The function will load human mobility data from the input file and
    convert it into a weighted undirected NetworkX Graph.
    Each node corresponds to a country represented by its 3-letter
    ISO 3166-1 alpha-3  code. Each edge between a pair of countries is
    weighted with the number of average daily trips between the two countries.
    The dataset contains annual trips for varying numbers of years, and the daily
    average is computed and stored below.
    """

    df = pd.read_csv(fname,header=0) #Read dataset into Pandas dataframe, may take 1-2 minutes


    #Convert dataframe into D, a dictionary of dictionaries
    #Each key is a country, and the corresponding value is
    #a dictionary which has a linked country as a key
    #and a 2-element list as a value.  The 2-element list contains
    #the total number of trips between two countries and the number of years
    #over which these trips were taken
    D = {}
    for index, row in df.iterrows():
         c1,c2,yr,N = row[0],row[1],row[2],row[3]
         if len(c1)<=3:
             if c1 not in D:
                 D[c1] = {c2:[N,1]}
             else:
                 if c2 not in D[c1]:
                     D[c1][c2] = [N,1]
                 else:
                     Nold,count = D[c1][c2]
                     D[c1][c2] = [N+Nold,count+1]


    #Create new dictionary of dictionaries which contains the average daily
    #number of trips between two countries rather than the 2-element lists
    #stored in D
    Dnew = {}
    for k,v in D.items():
        Dnew[k]={}
        for k2,v2 in v.items():
            if v2[1]>0:
                v3 = D[k2][k]
                w_ave = (v2[0]+v3[0])/(730*v2[1])
                if w_ave>0: Dnew[k][k2] = {'weight':w_ave}

    G = nx.from_dict_of_dicts(Dnew) #Create NetworkX graph

    return G


def network(G=data('project4.csv'),inputs=()):
    """
    Analyze input networkX graph, G
    Use inputs to provide any other needed information.
    """
    N=G.number_of_nodes()

    #create information for 1st histogram
    degrees_dict=set()
    weight_dict={}
    count_dict={}
    for i in G:
      j=G.degree()[i]
      if j in degrees_dict:
        count_dict[j]+=1
        weight_dict[j]+= G.degree(weight='weight')[i]
      else:
        count_dict[j]=1
        weight_dict[j]= G.degree(weight='weight')[i]
        degrees_dict.add(j)

    weight_list_ave=[]
    degrees_list=[]
    count_list=[]
    for i in range(N):
      if i in degrees_dict:
        weight_list_ave.append( max(np.log(weight_dict[i]/count_dict[i]),0))
        degrees_list.append(i)
        count_list.append(count_dict[i])

    #plot of first histogram
    fig, ax1 = plt.subplots(figsize =(14,7))

    ax1.bar(degrees_list,weight_list_ave, label='average node weight',color=['r'],alpha=0.6)
    #ax1.hist(nodes_degrees_list, bins=196, label=['Nt=10000'],color=['r'],alpha=0.6)
    ax1.set_xlabel('degree of node')
    ax1.set_ylabel('frequency')

    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.bar(degrees_list,count_list, label='node degrees',color=['b'],alpha=0.6)
    #ax2.hist(nodes_ave_weight_list, bins=196, label=['graph node frequency'],color=['b'],alpha=0.6)
    ax2.set_ylabel('Log(Sum of Weights/Frequency)')
    ax2.tick_params(axis='y')

    ax1.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.legend(loc="upper center", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('Node degrees and log of average node weight', y=1.01, fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()

    pos2=positions()

    pos=pos2
    #creates centrality measures
    measures={}
    for i in G:
      measures[i]=G.degree(weight='weight')[i]#/G.degree()[i]
    del measures["AND"]
    node_size1=120

    #plot graph
    plt.figure(figsize=(20, 13))
    all_weights = []
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size1,alpha=1, cmap=plt.cm.plasma,node_color=list(measures.values()),nodelist=list(measures.keys()),linewidths=0.00)
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    #nodes.set_norm(mcolors.Normalize())
    nx.draw_networkx_labels(G, pos,font_size=10)#,font_color='r')

    #Iterate through graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) # I use this when determining edge thickness

    #unique weights
    unique_weights = list(set(all_weights))
    max_weight=max(unique_weights)

    for weight in unique_weights:
        #add edges of signifcant weight
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        width = min(weight*35/max_weight,1)
        nx.draw_networkx_edges(G,pos2,edgelist=weighted_edges,width=width,edge_color='g')

    plt.title("Graph showing sum of weights going into nodes and significant edges")
    plt.colorbar(nodes)
    plt.show()

    ######## Code for figure 3 #######
    pos=pos2
    measures=nx.eigenvector_centrality(G,max_iter=800,weight='weight')
    node_size1=120

    plt.figure(figsize=(20, 13))
    all_weights = []
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size1,alpha=1, cmap=plt.cm.plasma,node_color=list(measures.values()),nodelist=list(measures.keys()),linewidths=0.00)
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    #nodes.set_norm(mcolors.Normalize())
    nx.draw_networkx_labels(G, pos,font_size=10)

    #Iterate through graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) # I use this when determining edge thickness

    #unique weights
    unique_weights = list(set(all_weights))
    max_weight=max(unique_weights)

    for weight in unique_weights:
        #add edges of signifcant weight
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        width = min(weight*30/max_weight,1)
        nx.draw_networkx_edges(G,pos2,edgelist=weighted_edges,width=width)

    plt.title("Graph showing eigenvector centrality and significant edges")
    plt.colorbar(nodes)
    plt.show()
    return None




def modelBH(G,x=0,i0=0.1,alpha=0.45,beta=0.3,gamma=1e-3,eps=1.0e-6,eta=8,tf=20,Nt=1000):
    """
    Simulate model Brockmann & Helbing SIR model

    Input:
    G: Weighted undirected Networkx graph
    x: node which is initially infected with j_x=i0
    i0: magnitude of initial condition
    alpha,beta,gamma,eps,eta: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    jarray: Nt+1 x N array containing j across the N network nodes at
                each time step.
    sarray: Nt+1 x N array containing s across network nodes at each time step
    """


    #convert keys to intergers not essential but easier to work with
    G = nx.convert_node_labels_to_integers(G,label_attribute='country')
    country_dict={}
    for i in range(len(G)):
      country_dict[G.nodes[i]['country']]=i

    #create numpy adjecnecy matrix
    A = nx.adjacency_matrix(G).todense()
    N = G.number_of_nodes()
    tarray = np.linspace(0,tf,Nt+1)
    #first N are j_n and 2nd are sn
    iarray = np.zeros((Nt+1,2*N))

    #create P, could be done more efficently but meh
    P=np.zeros((N,N))
    for i in range(N):
      P[i,:]=A[i,:]/(A[i,:].sum())

    #set initial conditions at the start everyone is suspetible apart from the populations already infected, assuming noones
    y0=np.zeros(2*N)
    y0[country_dict[x]]=i0
    y0[N:]=np.ones(N)
    y0[country_dict[x]+N]=1-i0

    #precomputations and seting up change array
    dy=np.zeros(N*2)
    eps_eta=eps**eta
    gamma_beta=gamma+beta

    def sigma(x):
      return x/(eps_eta+x)

    #bulk work of algorithm done inRHS
    def RHS(y,t):
        """Compute RHS of equation 3 at time t
        input: y should be a size 2*N array
        output: dy, also a size 2*N array corresponding to dy/dt

        Discussion: add discussion here
        """

        dy[:N] = alpha*y[:N]*y[N:2*N]*sigma(np.power(y[:N],eta))
        dy[N:] = gamma*(np.matmul(P,y[N:])-y[N:])-dy[:N]
        dy[:N] += gamma*np.matmul(P,y[:N])-gamma_beta*y[:N]
        return dy

    #calculate solution
    iarray[:,:]=odeint(RHS,y0,tarray)

    #extract solution
    jarray=iarray[:,:N]
    sarray=iarray[:,N:]

    return tarray,jarray,sarray


def analyze(G,inputs=()):
    """Compute effective distance matrix and
    analyze simulation results
    Input:
        G: Weighted undirected NetworkX graphs
        inputs: can be used to provide additional needed information
    Output:
        D: N x N effective distance matrix (a numpy array)

    """

    G1 = nx.convert_node_labels_to_integers(G,label_attribute='country')
    country_dict={}
    for i in range(len(G1)):
      country_dict[G1.nodes[i]['country']]=i
    A = nx.adjacency_matrix(G1).todense()
    N = G1.number_of_nodes()

    figures_for_analyze(G)

    P=np.zeros((N,N))
    for i in range(N):
      P[i,:]=A[i,:]/(A[i,:].sum())

    d=np.zeros((N,N))
    iK,jK = np.where( P != 0)
    for i,j in zip(iK,jK):
      d[i,j]=1-np.log(P[i,j])

    #create new directed graph using effective distance d
    G2=nx.from_numpy_matrix(d,create_using=nx.DiGraph)
    #find shortest paths
    D=nx.floyd_warshall_numpy(G2, weight="weight")
    figures_for_analyze(G,D)
    return D

def figures_for_analyze(G,D):
    """returns None, but makes all the figures foranalyze
    """

    Nt=10000
    tarray,jarray,sarray=modelBH(G,x='CHN',i0=0.1,alpha=0.45,beta=0.3,gamma=1e-3,eps=1.0e-6,eta=8,tf=200,Nt=Nt)
    plt.figure(figsize=(12, 6))
    plt.plot(tarray[::10],jarray[::10,:])
    plt.xlabel('time')
    plt.ylabel('j_n')
    plt.title("BH model for initial outbreak in China and i0=0.1,alpha=0.45,beta=0.4,gamma=1e-3,eps=1.0e-6,eta=8")
    plt.show()
    tarray,jarray,sarray=modelBH(G,x='PER',i0=0.1,alpha=0.45,beta=0.3,gamma=1e-3,eps=1.0e-6,eta=8,tf=200,Nt=Nt)
    plt.figure(figsize=(12, 6))
    plt.plot(tarray[::10],jarray[::10,:])
    plt.xlabel('time')
    plt.ylabel('j_n')
    plt.title("BH model for initial outbreak in Peru and i0=0.1,alpha=0.45,beta=0.4,gamma=1e-3,eps=1.0e-6,eta=8")
    plt.show()
    #create dictionary to convert country name to index of graph
    G1 = nx.convert_node_labels_to_integers(G,label_attribute='country')
    country_dict={}
    for i in range(len(G1)):
      country_dict[G1.nodes[i]['country']]=i
    #get positions
    pos=positions()

    #set start position
    x0='CHN'
    #obtain info from position to calulate distance and create ordered lists and indexs
    dist_dict={}
    for i in pos:
      dist_dict[np.linalg.norm(pos[i]-pos[x0])]=i

    dist_ind=[]
    dist_list=[]
    for i in sorted(dist_dict):
      dist_ind.append(int(country_dict[dist_dict[i]]))
      dist_list.append(i)

    #obtain info from effective distance and create ordered lists and indexs
    D=analyze(G)
    eff_dist=D[:,country_dict[x0]]
    eff_dist_dict={}
    for i in pos:
      j=country_dict[i]
      eff_dist_dict[float(eff_dist[int(j)])]=i

    eff_dist_ind=[]
    eff_dist_list=[]
    for i in sorted(eff_dist_dict):
      eff_dist_ind.append(int(country_dict[eff_dist_dict[i]]))
      eff_dist_list.append(i)

    Nt=10000
    tarray,jarray,sarray=modelBH(G,x='CHN',i0=0.1,tf=200,Nt=Nt)

    plt.figure(figsize=(10,12 ))
    plt.contour(eff_dist_list,tarray[::10],jarray[::10,eff_dist_ind],80)
    plt.xlabel('Effective distance from source node')
    plt.ylabel('t')
    plt.colorbar()
    plt.title('Contours of j_n of BH for inital outbreak of China using Effective distances ')

    plt.figure(figsize=(10,12 ))
    plt.contour(dist_list,tarray[::10],jarray[::10,dist_ind],80)
    plt.xlabel('Euclidean distance from source node')
    plt.ylabel('t')
    plt.colorbar()
    plt.title('Contours of j_n of BH for inital outbreak of China using Euclidean distances')

    #set start position
    x0='PER'
    #obtain info from position to calulate distance and create ordered lists and indexs
    dist_dict={}
    for i in pos:
      dist_dict[np.linalg.norm(pos[i]-pos[x0])]=i

    dist_ind=[]
    dist_list=[]
    for i in sorted(dist_dict):
      dist_ind.append(int(country_dict[dist_dict[i]]))
      dist_list.append(i)

    #obtain info from effective distance and create ordered lists and indexs
    eff_dist=D[:,country_dict[x0]]
    eff_dist_dict={}
    for i in pos:
      j=country_dict[i]
      eff_dist_dict[float(eff_dist[int(j)])]=i

    eff_dist_ind=[]
    eff_dist_list=[]
    for i in sorted(eff_dist_dict):
      eff_dist_ind.append(int(country_dict[eff_dist_dict[i]]))
      eff_dist_list.append(i)

    Nt=10000
    tarray,jarray,sarray=modelBH(G,x='PER',i0=0.1,alpha=0.45,beta=0.3,gamma=1e-3,eps=1.0e-6,eta=8,tf=200,Nt=Nt)

    plt.figure(figsize=(10,12 ))
    plt.contour(eff_dist_list,tarray[::10],jarray[::10,eff_dist_ind],80)
    plt.xlabel('Effective distance from source node')
    plt.ylabel('t')
    plt.colorbar()
    plt.title('Contours of j_n of BH for inital outbreak of Peru using effective distances ')

    plt.figure(figsize=(10,12 ))
    plt.contour(dist_list,tarray[::10],jarray[::10,dist_ind],80)
    plt.xlabel('Euclidean distance from source node')
    plt.ylabel('t')
    plt.colorbar()
    plt.title('Contours of j_n of BH for inital outbreak of Peru using Euclidean distances')

    return None










def positions():
    """ductionary of avergae latidtue and londitde values
    """
    def array(x):
      return np.array(x)

    #dictionary giving ave lat and longtiude of a given country, was very annoying to make
    pos2={'AFG': array([65., 33.]),
     'AGO': array([ 18.5, -12.5]),
     'ALB': array([20., 41.]),
     'AND': array([ 1.6, 42.5]),
     'ARE': array([54., 24.]),
     'ARG': array([-64., -34.]),
     'ARM': array([45., 40.]),
     'ATG': array([-61.8 ,  17.05]),
     'AUS': array([133., -27.]),
     'AUT': array([13.3333, 47.3333]),
     'AZE': array([47.5, 40.5]),
     'BDI': array([30. , -3.5]),
     'BEL': array([ 4.    , 50.8333]),
     'BEN': array([2.25, 9.5 ]),
     'BFA': array([-2., 13.]),
     'BGD': array([90., 24.]),
     'BGR': array([25., 43.]),
     'BHR': array([50.55, 26.  ]),
     'BHS': array([-76.  ,  24.25]),
     'BIH': array([18., 44.]),
     'BLR': array([28., 53.]),
     'BLZ': array([-88.75,  17.25]),
     'BMU': array([-64.75  ,  32.3333]),
     'BOL': array([-65., -17.]),
     'BRA': array([-55., -10.]),
     'BRB': array([-59.5333,  13.1667]),
     'BRN': array([114.6667,   4.5   ]),
     'BTN': array([90.5, 27.5]),
     'BWA': array([ 24., -22.]),
     'CAF': array([21.,  7.]),
     'CAN': array([-95.,  60.]),
     'CHE': array([ 8., 47.]),
     'CHL': array([-71., -30.]),
     'CHN': array([105.,  35.]),
     'CIV': array([-5.,  8.]),
     'CMR': array([12.,  6.]),
     'COD': array([25.,  0.]),
     'COG': array([15., -1.]),
     'COL': array([-72.,   4.]),
     'COM': array([ 44.25  , -12.1667]),
     'CPV': array([-24.,  16.]),
     'CRI': array([-84.,  10.]),
     'CUB': array([-80. ,  21.5]),
     'CYM': array([-80.5,  19.5]),
     'CYP': array([33., 35.]),
     'CZE': array([15.5 , 49.75]),
     'DEU': array([ 9., 51.]),
     'DJI': array([43. , 11.5]),
     'DMA': array([-61.3333,  15.4167]),
     'DNK': array([10., 56.]),
     'DOM': array([-70.6667,  19.    ]),
     'DZA': array([ 3., 28.]),
     'ECU': array([-77.5,  -2. ]),
     'EGY': array([30., 27.]),
     'ERI': array([39., 15.]),
     'ESP': array([-4., 40.]),
     'EST': array([26., 59.]),
     'ETH': array([38.,  8.]),
     'FIN': array([26., 64.]),
     'FJI': array([175., -18.]),
     'FRA': array([ 2., 46.]),
     'FSM': array([158.25  ,   6.9167]),
     'GAB': array([11.75, -1.  ]),
     'GBR': array([-2., 54.]),
     'GEO': array([43.5, 42. ]),
     'GHA': array([-2.,  8.]),
     'GIB': array([-5.3667, 36.1833]),
     'GIN': array([-10.,  11.]),
     'GMB': array([-16.5667,  13.4667]),
     'GNB': array([-15.,  12.]),
     'GNQ': array([10.,  2.]),
     'GRC': array([22., 39.]),
     'GTM': array([-90.25,  15.5 ]),
     'GUY': array([-59.,   5.]),
     'HKG': array([114.1667,  22.25  ]),
     'HND': array([-86.5,  15. ]),
     'HRV': array([15.5   , 45.1667]),
     'HTI': array([-72.4167,  19.    ]),
     'HUN': array([20., 47.]),
     'IDN': array([120.,  -5.]),
     'IND': array([77., 20.]),
     'IRL': array([-8., 53.]),
     'IRN': array([53., 32.]),
     'IRQ': array([44., 33.]),
     'ISL': array([-18.,  65.]),
     'ISR': array([34.75, 31.5 ]),
     'ITA': array([12.8333, 42.8333]),
     'JAM': array([-77.5 ,  18.25]),
     'JOR': array([36., 31.]),
     'JPN': array([138.,  36.]),
     'KAZ': array([68., 48.]),
     'KEN': array([38.,  1.]),
     'KGZ': array([75., 41.]),
     'KHM': array([105.,  13.]),
     'KIR': array([173.    ,   1.4167]),
     'KNA': array([-62.75  ,  17.3333]),
     'KOR': array([127.5,  37. ]),
     'KWT': array([47.6581, 29.3375]),
     'LAO': array([105.,  18.]),
     'LBN': array([35.8333, 33.8333]),
     'LBR': array([-9.5,  6.5]),
     'LBY': array([17., 25.]),
     'LCA': array([-61.1333,  13.8833]),
     'LKA': array([81.,  7.]),
     'LSO': array([ 28.5, -29.5]),
     'LTU': array([24., 56.]),
     'LUX': array([ 6.1667, 49.75  ]),
     'LVA': array([25., 57.]),
     'MAC': array([113.55  ,  22.1667]),
     'MAR': array([-5., 32.]),
     'MDA': array([29., 47.]),
     'MDG': array([ 47., -20.]),
     'MDV': array([73.  ,  3.25]),
     'MEX': array([-102.,   23.]),
     'MHL': array([168.,   9.]),
     'MKD': array([22.    , 41.8333]),
     'MLI': array([-4., 17.]),
     'MLT': array([14.5833, 35.8333]),
     'MMR': array([98., 22.]),
     'MNG': array([105.,  46.]),
     'MOZ': array([ 35.  , -18.25]),
     'MRT': array([-12.,  20.]),
     'MUS': array([ 57.55  , -20.2833]),
     'MWI': array([ 34. , -13.5]),
     'MYS': array([112.5,   2.5]),
     'NAM': array([ 17., -22.]),
     'NER': array([ 8., 16.]),
     'NGA': array([ 8., 10.]),
     'NIC': array([-85.,  13.]),
     'NIU': array([-129,  -19.0333]),
     'NLD': array([ 5.75, 52.5 ]),
     'NOR': array([10., 62.]),
     'NPL': array([84., 28.]),
     'NRU': array([166.9167,  -0.5333]),
     'NZL': array([174., -41.]),
     'OMN': array([57., 21.]),
     'PAK': array([70., 30.]),
     'PAN': array([-80.,   9.]),
     'PER': array([-76., -10.]),
     'PHL': array([122.,  13.]),
     'PLW': array([134.5,   7.5]),
     'PNG': array([147.,  -6.]),
     'POL': array([20., 52.]),
     'PRK': array([127.,  40.]),
     'PRT': array([-8. , 39.5]),
     'PRY': array([-58., -23.]),
     'PSE': array([35.25, 32.  ]),
     'QAT': array([51.25, 25.5 ]),
     'ROU': array([25., 46.]),
     'RUS': array([100.,  60.]),
     'RWA': array([30., -2.]),
     'SAU': array([45., 25.]),
     'SDN': array([30., 15.]),
     'SEN': array([-14.,  14.]),
     'SGP': array([103.8   ,   1.3667]),
     'SLB': array([159.,  -8.]),
     'SLE': array([-11.5,   8.5]),
     'SLV': array([-88.9167,  13.8333]),
     'SMR': array([12.4167, 43.7667]),
     'SOM': array([49., 10.]),
     'STP': array([7., 1.]),
     'SUR': array([-56.,   4.]),
     'SVK': array([19.5   , 48.6667]),
     'SVN': array([15., 46.]),
     'SWE': array([15., 62.]),
     'SWZ': array([ 31.5, -26.5]),
     'SYC': array([55.6667, -4.5833]),
     'SYR': array([38., 35.]),
     'TCA': array([-71.5833,  21.75  ]),
     'TCD': array([19., 15.]),
     'TGO': array([1.1667, 8.    ]),
     'THA': array([100.,  15.]),
     'TJK': array([71., 39.]),
     'TKM': array([60., 40.]),
     'TLS': array([125.5167,  -8.55  ]),
     'TON': array([-130.,  -20.]),
     'TTO': array([-61.,  11.]),
     'TUN': array([ 9., 34.]),
     'TUR': array([35., 39.]),
     'TUV': array([178.,  -8.]),
     'TZA': array([35., -6.]),
     'UGA': array([32.,  1.]),
     'UKR': array([32., 49.]),
     'URY': array([-56., -33.]),
     'USA': array([-97.,  38.]),
     'UZB': array([64., 41.]),
     'VCT': array([-61.2 ,  13.25]),
     'VEN': array([-66.,   8.]),
     'VGB': array([-64.5,  18.5]),
     'VNM': array([106.,  16.]),
     'VUT': array([167., -16.]),
     'WSM': array([-128.3333,  -13.5833]),
     'YEM': array([48., 15.]),
     'ZAF': array([ 24., -29.]),
     'ZMB': array([ 30., -15.]),
     'ZWE': array([ 30., -20.])}
    return pos2




if __name__=='__main__':
    #Add code below to call network and analyze so that they generate the figures
    #in your report.
    G1=data('project4.csv')
    network(G=G1)
    analyze(G=G1)
    return None
