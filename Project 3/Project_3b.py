"""Scientific Computation Project 2, part 2
Your CID here: 1059624
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy


def rwgraph(G,i0=0,M=100,Nt=100):
    """ Question 2.1
    Simulate M Nt-step random walks on input graph, G, with all
    walkers starting at node i0
    Input:
        G: An undirected, unweighted NetworkX graph
        i0: intial node for all walks
        M: Number of walks
        Nt: Number of steps per walk
    Output: X: M x Nt+1 array containing the simulated trajectories
    """

    adj_dict=nx.convert.to_dict_of_lists(G)
    X=np.ones((M,Nt+1),dtype=int)
    X[:,0]=X[:,0]*i0

    n=len(adj_dict)

    if M*Nt<n:
        for i in range(M):
            for j in range(0,Nt):
                X[i,j+1]=np.random.choice(adj_dict[X[i,j]])

    else:
        #creates array of degree of nodes
        degree_arr=np.asarray(G.degree(),dtype=int)[:,1]
        max_degree=max(degree_arr)

        #create denses matrix
        Mat=np.zeros((n,max_degree+1))
        for i in range(0,n):
          Mat[i,0:len(adj_dict[i])]= np.asarray(adj_dict[i],dtype=int)

        w=np.random.random((M,Nt))
        #iterate through time
        for j in range(0,Nt):
            X[:,j+1]=Mat[X[:,j],np.floor(degree_arr[X[:,j]]*w[:,j]).astype(int)]
    return X


def rwgraph_analyze1(input=(None)):
    """Analyze simulated random walks on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    #generates graph
    n=2000
    m=4
    G=nx.barabasi_albert_graph(n, m, seed=5)

    Nt=100
    M=20000
    #finds max degree of graph and stores list of degrees of nodes
    maxdeg=0
    degree_dist=[]
    for i in range(0,n):
      degree_dist.append(G.degree[i])
      if G.degree[i]>maxdeg:
        maxdeg=G.degree[i]
        j=i
    #generates data and stores them in lists for varyin M and Nt
    X=rwgraph(G,j,M,Nt)
    Listnodes=[]
    for i in range(M):
      Listnodes.append(G.degree(X[i,Nt]))
    Nt=10000
    M=20000
    X=rwgraph(G,j,M,Nt)
    Listnodes2=[]
    for i in range(M):
      Listnodes2.append(G.degree(X[i,Nt]))
    Nt=10
    M=20000
    X=rwgraph(G,j,M,Nt)
    Listnodes3=[]
    for i in range(M):
      Listnodes3.append(G.degree(X[i,Nt]))
    Nt=10000
    M=200
    X=rwgraph(G,j,M,Nt)
    Listnodes4=[]
    for i in range(M):
      Listnodes4.append(G.degree(X[i,Nt]))
    fig, ax1 = plt.subplots(figsize =(14,7))

    ##### creates histo gram figure with 2 axis####
    ax1.hist([Listnodes,Listnodes2], bins=maxdeg, label=['Nt=100', 'Nt=10000'],color=['g','r'],alpha=0.6)
    ax1.set_xlabel('degree of node')
    ax1.set_ylabel('frequency of final position of random walks')

    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.hist([degree_dist], bins=maxdeg, label=['graph node frequency'],color=['b'],alpha=0.6)
    ax2.set_ylabel('frequency of node degrees for graph')
    ax2.tick_params(axis='y')

    ax1.legend(loc="center right", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.legend(loc="upper right", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('M=20000, node degree of final position of random walk, for varying amounts of time', y=1.10, fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()

    #function to generate diction of frequency
    def CountFrequency(my_list):

        # Creating an empty dictionary
        freq = {}
        for item in my_list:
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1
        return freq
    #converts data to approprate form so it can plotted on scatter plot
    #frequecy
    listfreq1=CountFrequency(Listnodes2)
    listfreq2=CountFrequency(Listnodes3)
    listfreq3=CountFrequency(Listnodes4)
    listfreq_deg=CountFrequency(degree_dist)
    #set up lists
    z=[]
    z2=[]
    z3=[]
    z_deg=[]
    z_deg2=[]
    z_deg3=[]
    #code to create list of only degrees used in simulations
    for i in listfreq1:
      z.append(listfreq1[i]/(listfreq_deg[i]*20000))
      z_deg.append(i)
    for i in listfreq2:
      z2.append(listfreq2[i]/(listfreq_deg[i]*20000))
      z_deg2.append(i)
    for i in listfreq3:
      z3.append(listfreq3[i]/(listfreq_deg[i]*200))
      z_deg3.append(i)
    #extpected prob distribution
    E=G.number_of_edges()
    z0=[]
    z_deg0=[]
    for i in listfreq_deg:
      z0.append(i/(2*E))
      z_deg0.append(i)
    #genrates scatter plot figure
    plt.figure(figsize=(12, 6))
    plt.scatter(z_deg, z, label='Nt=10000, M=20000')
    plt.scatter(z_deg2, z2,label='Nt=10, M=20000')
    plt.scatter(z_deg3, z3,label='Nt=10, M=200')
    plt.plot(z_deg0,z0,label="expected prob dist",alpha=0.5)
    plt.xlabel('degree of node')
    plt.ylabel('frequency of final position / M*frequency of degree')
    plt.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title("Frequency of final positions relative to number of nodes of that degree, for changing times Nt and M.")
    plt.show()
    return None #modify as needed


def rwgraph_analyze2(input=(None)):
    """Analyze similarities and differences
    between simulated random walks and linear diffusion on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """


        #set up graph and degree distribution arrays
    n=2000
    m=4
    G=nx.barabasi_albert_graph(n, m, seed=5)
    Nt=100
    M=20000
    maxdeg=0
    degree_dist=[]
    for i in range(0,n):
      degree_dist.append(G.degree[i])
      if G.degree[i]>maxdeg:
        maxdeg=G.degree[i]
        j=i

    #set inital conditions and D
    y0=np.zeros(n,dtype=int)
    y0[j]=200
    D=1
    #define time for odi Int
    t=np.arange(Nt+1,dtype=int)
    #set up operators
    A = nx.adjacency_matrix(G)
    Q = A.toarray().sum(axis=1)
    L=np.diag(Q)-A.toarray()
    Q_inv=1/Q
    Ls=np.diag(np.ones(n))-np.matmul(np.diag(Q_inv),A.toarray())
    Ls_tran=np.transpose(Ls)

    #convert to sparse operators and include diffusion
    L_spar = scipy.sparse.csr_matrix(-D*L)
    Ls_spar = scipy.sparse.csr_matrix(-D*Ls)
    Ls_tran_spar = scipy.sparse.csr_matrix(-D*Ls_tran)
    A=nx.adjacency_matrix(G)
    L=-D*(scipy.sparse.diags(degree_arr)-A)
    Ls=-D*(scipy.sparse.diags(np.ones(N))-scipy.sparse.diags(1/degree_arr).dot(A))

    #define operators
    def Lap(y,t):
      return scipy.sparse.csr_matrix.__mul__(L_spar,y)
    def Lap_Ls(y,t):
      return scipy.sparse.csr_matrix.__mul__(Ls_spar,y)
    def Lap_Ls_tran(y,t):
      return scipy.sparse.csr_matrix.__mul__(Ls_tran_spar,y)

    #solutions of different operators
    solL=scipy.integrate.odeint(Lap,y0,t)
    solLs=scipy.integrate.odeint(Lap_Ls,y0,t)
    solLs_tran=scipy.integrate.odeint(Lap_Ls_tran,y0,t)


    #finds eigen values and vectors and puts them into order
    def eigen(L):
      eigen_values,eigen_vectors=scipy.linalg.eig(-L)
      idx = eigen_values.argsort()[::-1]
      eigen_values = eigen_values[idx]
      eigen_vectors = eigen_vectors[:,idx]
      return eigen_values,eigen_vectors

    #finds all eigen values and eigen vectors of the different operators. can use sparse matrics
    eigen_values_LS,eigen_vectors_LS=eigen(Ls)
    eigen_values_LS_tran,eigen_vectors_LS_tran=eigen(Ls_tran)
    eigen_values_L,eigen_vectors_L=eigen(L)
    eigen_values_L2,eigen_vectors_L2=eigen(L*0.36)

    ### could have eigs here as didn't end up using all eigenvalues ####
    #eigen values graph
    n0=len(eigen_values_L)
    eig_nums=np.arange(n0)
    plt.figure(figsize=(12, 6))
    plt.scatter(eig_nums[0:10],eigen_values_L2[0:10],s=50,marker="x" ,label='L , D=0.36')
    plt.scatter(eig_nums[0:10],eigen_values_LS[0:10],s=50, marker="|",label='LS , D=1')
    plt.scatter(eig_nums[0:10],eigen_values_LS_tran[0:10],s=50,marker='_',label='LS_tran , D=1')
    plt.scatter(eig_nums[0:10],eigen_values_L[0:10],s=50,marker="+" ,label='L , D=1')
    plt.legend(loc="lower left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel('eigen value number')
    plt.ylabel('eigenvalue')
    plt.title("Eigenvlaues of Laplacian Matrixs")
    plt.show()

    print("4 biggest eigenvalues for each operater")
    print('L=',eigen_values_L[0:4])
    print('Ls=',eigen_values_LS[0:4])
    print('Ls_tran=',eigen_values_LS_tran[0:4])
    #prints 4 biggest eigen values
    #counts node distrubtion by creating dictionary
    def result_count(sol,Nt,G):
        """ returns cumlative frequency/probailties for nodes of same degree and returns dictionary"""
        n = G.number_of_nodes()
        dict_freq={}
        for i in range(n):
          k=G.degree(i)
          if k not in dict_freq:
            dict_freq[k]=sol[Nt,i]
          else:
            dict_freq[k]+=sol[Nt,i]
        return dict_freq

    #frequency count of solutions
    dict_freq=result_count(solL,Nt,G)
    dict_freq2=result_count(solLs,Nt,G)
    dict_freq3=result_count(solLs_tran,Nt,G)

    #random walk data
    X=rwgraph(G,j,20000,100)
    Listnodes7=[]
    for i in range(20000):
      Listnodes7.append(G.degree(X[i,100]))
    X=rwgraph(G,j,200,100)
    Listnodes8=[]
    for i in range(200):
      Listnodes8.append(G.degree(X[i,100]))
    X=rwgraph(G,j,50000,5000)
    Listnodes9=[]
    for i in range(50000):
      Listnodes9.append(G.degree(X[i,5000]))
    listfreq7=CountFrequency(Listnodes7)
    listfreq8=CountFrequency(Listnodes8)
    listfreq9=CountFrequency(Listnodes9)
    listfreq_deg=CountFrequency(degree_dist)
    z2=[]
    z3=[]
    z1=[]
    z_deg2=[]
    z_deg3=[]
    z_deg1=[]
    for i in listfreq7:
      z2.append(listfreq7[i]/(listfreq_deg[i]*20000))
      z_deg2.append(i)
    for i in listfreq8:
      z3.append(listfreq8[i]/(listfreq_deg[i]*200))
      z_deg3.append(i)
    for i in listfreq8:
      z1.append(listfreq9[i]/(listfreq_deg[i]*50000))
      z_deg1.append(i)
    #operator solutions compared to node degree frequency
    z4,z5,z6=[],[],[]
    z_deg4,z_deg5,z_deg6=[],[],[]
    for i in dict_freq:
      z4.append(dict_freq[i]/(listfreq_deg[i]*200))
      z_deg4.append(i)
    for i in dict_freq2:
      z5.append(dict_freq2[i]/(listfreq_deg[i]*200))
      z_deg5.append(i)
    for i in dict_freq3:
      z6.append(dict_freq3[i]/(listfreq_deg[i]*200))
      z_deg6.append(i)

    plt.figure(figsize=(15, 10))
    plt.scatter(z_deg1, z1,label='Nt=5000, M=50000')
    plt.scatter(z_deg2, z2,label='Nt=100, M=20000')
    plt.scatter(z_deg3, z3,label='Nt=100, M=200')
    plt.scatter(z_deg4, z4,label='L, Nt=100')
    plt.scatter(z_deg5, z5,label='Ls, Nt=100')
    plt.scatter(z_deg6, z6,label='Ls_tran, Nt=100')
    plt.ylim((-0.005,0.020))
    plt.xlabel('degree of node')
    plt.ylabel('frequency of final position / M*frequency of degree')
    plt.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title("Frequency of final positions relative to number of nodes of that degree, for changing times Nt and M.")
    plt.show()

    #code to produce final graph
    iarray1=LinearModel(G,x=j,i0=1,L1='L',D=1,tf=20,Nt=Nt)
    iarray2=LinearModel(G,x=j,i0=1,L1='Ls',D=1,tf=20,Nt=Nt)
    iarray3=LinearModel(G,x=j,i0=1,L1='Lst',D=1,tf=20,Nt=Nt)
    tarray = np.linspace(0,5,Nt+1)
    plt.figure(figsize=(12, 6))
    plt.plot(tarray, iarray1[:,7] ,label='rand node L,deg=46',color='b',alpha=0.5)
    plt.plot(tarray, iarray2[:,7] ,label='rand node Ls,deg=46',marker='|',color='r')
    plt.scatter(tarray, iarray3[:,7] ,label='rand node LST,deg=46',marker='_',color='y')
    plt.scatter(tarray, iarray1[:,1801] ,label='rand node L, deg=5',color='m',alpha=0.5,marker='+')
    plt.plot(tarray, iarray2[:,1801] ,label='rand node Ls,deg=5',marker='|',color='c')
    plt.scatter(tarray, iarray3[:,1801] ,label='rand node LST,deg=5',marker='_',color='g')
    plt.xlabel('time')
    plt.ylabel('representive frequency')
    plt.legend()
    plt.title("Comparing repestive frequency of a random nodes, for the different linear models,time step=50,D=0.1")
    plt.show()
    return None #modify as needed



def modelA(G,x=0,i0=0.1,beta=1.0,gamma=1.0,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model A

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    beta,gamma: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
                each time step.
    """

    N = G.number_of_nodes()
    iarray = np.zeros((N,Nt+1))
    tarray = np.linspace(0,tf,Nt+1)
    A=(nx.adjacency_matrix(G))*gamma
    ones=np.ones(N)
    y0=np.zeros(N)
    y0[x]=i0


    def RHS(y,t):
        """Compute RHS of modelA at time t
        input: y should be a size N array
        output: dy, also a size N array corresponding to dy/dt

        Discussion: add discussion here
        """

        return np.multiply(A.dot(y),ones-y)-beta*y

    iarray[:,:]=np.transpose(scipy.integrate.odeint(RHS,y0,tarray))


    return iarray

def modelB(G,x=0,i0=0.1,alpha=-0.01,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model B

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    alpha: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: NxNt+1 Array containing i across network nodes at
                each time step.
    s: NxNt+1 value of s at every time step
    """
  #set up graph atteributes
    N = G.number_of_nodes()
    degree_arr=np.asarray(G.degree(),dtype=int)[:,1]
    iarray = np.zeros((Nt+1,2*N))
    tarray = np.linspace(0,tf,Nt+1)
    #calucalte operaters and set intial conditions
    A=nx.adjacency_matrix(G)
    L=scipy.sparse.diags(degree_arr)-A
    L_alpha=L*alpha
    ones=np.ones(2*N)

    y0=np.zeros(2*N)
    y0[N+x]=i0
    #Add code here
    dy=np.zeros(N*2)
    def RHS2(y,t):
        """Compute RHS of modelB at time t
        input: y should be a size N array
        output: dy, also a size N array corresponding to dy/dt

        Discussion: add discussion here
        """
        dy[:N] =y[N:2*N]
        dy[N:2*N]=scipy.sparse.csr_matrix.__mul__(L_alpha,y[0:N])
        return dy

    iarray[:,:]=scipy.integrate.odeint(RHS2,y0,tarray)

    return iarray[:,N:],iarray[:,:N]

def LinearModel(G,x=0,i0=0.1,L1='L',D=-0.01,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model linear models

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    L1: which laplcacian to use options are "L,Ls,Lst"
    D: diffusive parameter >0
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
                each time step.
    """
  #set up graph atteributes
    N = G.number_of_nodes()
    degree_arr=np.asarray(G.degree(),dtype=int)[:,1]
    iarray = np.zeros((Nt+1,N))
    tarray = np.linspace(0,tf,Nt+1)
    #calucalte operaters and set intial conditions
    A=nx.adjacency_matrix(G)
    L=-D*(scipy.sparse.diags(degree_arr)-A)
    Ls=-D*(scipy.sparse.diags(np.ones(N))-scipy.sparse.diags(1/degree_arr).dot(A))

    y0=np.zeros(N)
    y0[x]=i0
    #set up operators

    if L1=='Ls':
      L=Ls
    elif L1=='Lst':
      L=Ls.transpose()

    #define operators
    def Lap(y,t):
      return scipy.sparse.csr_matrix.__mul__(L,y)

    iarray[:,:]=scipy.integrate.odeint(Lap,y0,tarray)

    return iarray



def transport(input=(None)):
    """Analyze transport processes (model A, model B, linear diffusion)
    on Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    n=100
    m=5
    G=nx.barabasi_albert_graph(n, m, seed=5)
    maxdeg=0
    degree_dist=[]
    for i in range(0,n):
      degree_dist.append(G.degree[i])
      if G.degree[i]>maxdeg:
        maxdeg=G.degree[i]
        j=i
    tf,tfa,tfb=10,20,1000
    Nt=10000
    iarray=LinearModel(G,x=j,i0=1,L1='L',D=0.1,tf=tf,Nt=1000)
    iarrayA=np.transpose(modelA(G,x=j,i0=1,beta=0.5,gamma=0.1,tf=tfa,Nt=Nt))
    iarrayB,s=modelB(G,x=j,i0=1,alpha=-0.01,tf=tfb,Nt=Nt)
    tarray = np.linspace(0,tf,1000+1)
    tarraya = np.linspace(0,tfa,Nt+1)
    tarrayb = np.linspace(0,tfb,Nt+1)

    plt.figure(figsize=(12, 6))
    plt.plot(tarray,iarray[:,j+1:])
    plt.xlabel('time')
    plt.ylabel('Intensity')
    plt.title("Linear model for BA graph(n=100,m=5), D=0.1, with highest node omitted, time step=10")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(tarraya,iarrayA)
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title("Model A for BA graph(n=100,m=5), with beta=0.5,gamma=0.1,time step=20")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(tarrayb,iarrayB)
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title("Model B for BA graph(n=100,m=5), with alpha=-0.01, timestep=500")
    plt.show()

    #genrate data for tf=20 for all models
    tf=20
    iarray=LinearModel(G,x=j,i0=1,L1='L',D=0.1,tf=tf,Nt=Nt)
    iarrayA=np.transpose(modelA(G,x=j,i0=1,beta=0.5,gamma=0.1,tf=tf,Nt=Nt))
    iarrayB,s=modelB(G,x=j,i0=1,alpha=-0.01,tf=tf,Nt=Nt)
    tarray = np.linspace(0,tf,Nt+1)
    #generate the means
    mean=np.mean(iarray,axis=1)
    meanA=np.mean(iarrayA,axis=1)
    meanB=np.mean(iarrayB,axis=1)
    #generate thevar info
    var=np.var(iarray,axis=1)
    varA=np.var(iarrayA,axis=1)
    varB=np.var(iarrayB,axis=1)



    plt.figure(figsize=(12, 6))
    plt.plot(tarray, meanA ,label='Model A',color='r')
    plt.scatter(tarray, meanB ,label='Model B',marker="|" ,alpha=0.5)
    plt.scatter(tarray, mean ,label='Linear L ',marker="_")
    plt.xlabel('time')
    plt.ylabel('Mean Intensity for different models for BA graph(n=100,m=5)')
    plt.legend()
    plt.title("How Mean changes for different models for BA graph(n=100,m=5)")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(tarray, var ,label='Linear L')
    plt.plot(tarray, varA ,label='Model A')
    plt.plot(tarray, varB ,label='Model B')
    plt.xlabel('time')
    plt.ylabel('Var Intensity ')
    plt.legend()
    plt.title("How variance changes for different models of BA graphs (n=100,m=5)")
    plt.show()



    return None #modify as needed



#function to generate diction of frequency
def CountFrequency(my_list):
    """ used to count frequency of results in a list, returning dictionaary"""
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

def result_count(sol,Nt,G):
    """ returns cumlative frequency/probailties for nodes of same degree and returns dictionary"""
    n = G.number_of_nodes()
    dict_freq={}
    for i in range(n):
      k=G.degree(i)
      if k not in dict_freq:
        dict_freq[k]=sol[Nt,i]
      else:
        dict_freq[k]+=sol[Nt,i]
    return dict_freq

if __name__=='__main__':
    #add code here to call diffusion and generate figures equivalent
    #to those you are submitting
    G=None #modify as needed
