"""Scientific Computation Project 2, part 1
Your CID here: 1059624
"""


def flightLegs(Alist,start,dest):
    """
    Question 1.1
    Find the minimum number of flights required to travel between start and dest,
    and  determine the number of distinct routes between start and dest which
    require this minimum number of flights.
    Input:
        Alist: Adjacency list for airport network
        start, dest: starting and destination airports for journey.
        Airports are numbered from 0 to N-1 (inclusive) where N = len(Alist)
    Output:
        Flights: 2-element list containing:
        [the min number of flights between start and dest, the number of distinct
        jouneys with this min number]
        Return an empty list if no journey exist which connect start and dest
    """
    #set up lists
    N=len(Alist)
    L1 = [0]*N #Labels
    L2 = [-1]*N #Distances

    #set up deque
    Q=collections.deque([])
    Q.append(start)
    L1[start]=1
    L2[start]=0
    #set up L3 which is rolling sum of number of disctint paths to a given node
    L3 = L1

    #while loop exits once a dest node has been probed/added to que
    while L3[dest]==0:
        if len(Q)==0:
            return []
        x = Q.popleft()
        for v in Alist[x]:
            if L1[v]==0:
                Q.append(v) #add unexplored neighbors to back of queue
                L1[v]=1
                L2[v]=1+L2[x]
                L3[v]=L3[x]
            #check if path is minumm distance to node
            elif L2[x]+1==L2[v]:
                #alter number of distinct paths
                L3[v]+=L3[x]
    #stop duplicates scores
    L3[dest]=0
    #added up rolling sum score
    for v in Alist[dest]:
        if L2[dest]-1==L2[v]:
            L3[dest]+=L3[v]
    #return desired list
    return [L2[dest],L3[dest]]


def safeJourney(Alist,start,dest):
    """
    Question 1.2 i)
    Find safest journey from station start to dest
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the density for the connection.
    start, dest: starting and destination stations for journey.

    Output:
        Slist: Two element list containing safest journey and safety factor for safest journey
    """
    N=len(Alist)
    #Initialize dictionaries
    dinit = float('inf')
    Edict = {} #Explored nodes
    Udict = {} #Unexplored nodes

    for i in range(0,N):
        Udict[i] = (dinit,-1)
    Udict[start]=(0,0)
    #Main search
    while dest not in Edict:
        #Find node with min d in Udict and move to Edict
        dmin = dinit
        for n,w in Udict.items():
            if w[0]<dmin:
                dmin=w[0]
                nmin=n
        if dmin==dinit:
            return []
        Edict[nmin] = Udict.pop(nmin)
        #print("moved node", nmin)

        #Update provisional distances for unexplored neighbors of nmin
        for n,w in Alist[nmin]:
            if n in Edict:
                pass
            elif w<Udict[n][0]:
                Udict[n]=(max(w,dmin),nmin)

    Q=collections.deque([dest])
    i = dest
    while i!=0:
        i=Edict[i][1]
        Q.appendleft(i)
    return [list(Q),Edict[dest][0]]

def shortJourney(Alist,start,dest):
    """
    Question 1.2 ii)
    Find shortest journey from station start to dest. If multiple shortest journeys
    exist, select journey which goes through the smallest number of stations.
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the time for the connection (rounded to the nearest minute).
    start, dest: starting and destination stations for journey.

    Output:
        Slist: Two element list containing shortest journey and duration of shortest journey
    """
    N=len(Alist)
    #Initialize dictionaries
    dinit = float('inf')
    Edict = {} #Explored nodes
    Udict = {} #Unexplored nodes

    for i in range(0,N):
        Udict[i] = [dinit,-1,-1]
    Udict[start]=[0,0,0]

    nmin=dest+1
    #Main search
    while nmin!= dest:
        #Find node with min d in Udict and move to Edict
        dmin = dinit
        for n,w in Udict.items():
            if w[0]<dmin:
                dmin=w[0]
                nmin=n
        if dmin==dinit:
            return []
        Edict[nmin] = Udict.pop(nmin)
        #Update provisional distances for unexplored neighbors of nmin
        for n,w in Alist[nmin]:
            if n in Edict:
                pass
            elif dmin+w<Udict[n][0]:
                Udict[n]=[dmin+w,nmin,Edict[nmin][2]+1]
            elif dmin+w==Udict[n][0] and Udict[n][2]>Edict[nmin][2]+1:
                Udict[n][1:3] = [nmin,Edict[nmin][2]+1]

    Q=collections.deque([dest])
    i = dest
    while i!=0:
        i=Edict[i][1]
        Q.appendleft(i)
    return [list(Q),Edict[dest][0]]



def cheapCycling(SList,CList):
    """
    Question 1.3
    Find first and last stations for cheapest cycling trip
    Input:
        Slist: list whose ith element contains cheapest fare for arrival at and
        return from the ith station (stored in a 2-element list or tuple)
        Clist: list whose ith element contains a list of stations which can be
        cycled to directly from station i
    Stations are numbered from 0 to N-1 with N = len(Slist) = len(Clist)
    Output:
        stations: two-element list containing first and last stations of journey
    """

    #Add code here


    N=len(Slist)
    #create dictionary of unexplored nodes
    Udict=set()
    for i in range(N):
        Udict.add(i)
    #pre set the best total
    best_tot=float('inf')
    Q=collections.deque([])

    #iterate until all nodes have been explored
    while len(Udict)>0:
        #select random unexplored node
        w=next(iter(Udict))
        Udict.remove(w)

        #removes solitory nodes
        if Clist[w]==[]:
            continue
        #que for BFS
        Q.append(w)
        #sets intial minum conditions
        x_arr1=[Slist[w][0],w]
        x_arr2=[float('inf'),-1]
        x_lev1=[Slist[w][1],w]
        x_lev2=[float('inf'),-1]

        #use BFS on Q, updating 1st and 2nd best fares  fo leaving and arring
        while len(Q)>0:
            x = Q.popleft()
            for v in Clist[x]:
                if v in Udict:

                    Q.append(v) #add unexplored neighbors to back of queue
                    Udict.remove(v) #explored node remove from set
                    y,z=Slist[v]
                    #check for if better then 2nd best minumum arriving fare of a connected tree
                    if y<x_arr1[0]:
                        x_arr2=x_arr1
                        x_arr1=[y,v]
                    elif y<x_arr2[0]:
                        x_arr2=[y,v]
                    #check leaving fare
                    if z<x_lev1[0]:
                        x_lev2=x_lev1
                        x_lev1=[z,v]
                    elif z<x_lev2[0]:
                        x_lev2=[z,v]

        #find best pair of nodes and then compares it to current best usually will exit after first if loop
        if x_arr1[1]!=x_lev1[1]:
            tot=x_arr1[0]+x_lev1[0]
            if tot<best_tot:
                best_tot=tot
                x_a_best=x_arr1[1]
                x_l_best=x_lev1[1]

        elif x_arr2[0]-x_arr1[0]<x_lev2[0]-x_lev1[0]:
            tot=x_arr2[0]+x_lev1[0]
            if tot<best_tot:
                best_tot=tot
                x_a_best=x_arr2[1]
                x_l_best=x_lev1[1]
        else:
            tot=x_arr1[0]+x_lev2[0]
            if tot<best_tot:
                best_tot=tot
                x_a_best=x_arr1[1]
                x_l_best=x_lev2[1]

    return [x_a_best,x_l_best]





if __name__=='__main__':
    #add code here if/as desired
    L=None #modify as needed
