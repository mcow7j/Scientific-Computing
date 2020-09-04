"""Scientific Computation Project 3, part 2
Your CID here 01059624
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy

def microbes(phi,kappa,mu,L = 1024,Nx=1024,Nt=1201,T=600,display=False,extraoutputs=False):
    """
    Question 2.2
    Simulate microbe competition model

    Input:
    phi,kappa,mu: model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    def RHS(y,t,k,r,phi,dx2inv):
        #RHS of model equations used by odeint

        n = y.size//2

        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        R = f/(f+phi)
        dfdt = d2f + f[1:-1]*(1-f[1:-1])- R[1:-1]*g[1:-1]
        dgdt = d2g - r*k*g[1:-1] + k*R[1:-1]*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = -4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy


    #Steady states
    rho = mu/kappa
    F = rho*phi/(1-rho)
    G = (1-F)*(F+phi)
    y0 = np.zeros(2*Nx) #initialize signal
    y0[:Nx] = F
    y0[Nx:] = G + 0.01*np.cos(10*np.pi/L*x) + 0.01*np.cos(20*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    print("running simulation...")
    y = odeint(RHS,y0,t,args=(kappa,rho,phi,dx2inv),rtol=1e-6,atol=1e-6)
    f = y[:,:Nx]
    g = y[:,Nx:]
    print("finished simulation")
    if display:
        plt.figure(figsize=(10,12 ))
        plt.contour(x,t[-(Nt%739):-1],f[-(Nt%739):-1,:])
        plt.xlabel('x')
        plt.ylabel('t')
        plt.colorbar()
        plt.title('Contours of f')
        plt.figure(figsize=(10,12))
        plt.contour(x,t[-(Nt%739):-1],g[-(Nt%739):-1,:])
        plt.xlabel('x')
        plt.ylabel('t')
        plt.colorbar()
        plt.title('Contours of g')
    if extraoutputs:
      return f,g,t,x

    return f,g

def newdiff(f,h):
    """
    Question 2.1 i)
    Input:
        f: array whose 2nd derivative will be computed
        h: grid spacing
    Output:
        d2f: second derivative of f computed with compact fd scheme
    """
    ####all of the following could be precomputed outside of function for all time ######
    N = len(f)
    #Coefficients for compact fd scheme
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140
    b=b/4
    c=c/9
    #set up banded matrix form ab of triadiagaonl system
    ab = np.ones((3,N))
    ab[0,:] = ab[0,:]*alpha
    ab[2,:] = ab[2,:]*alpha
    ab[0,0:2] = [0,10]
    ab[2,-1] = 0
    ab[2,-2] = 10
    #coefficents for ends solutions
    g = np.array([145/12,-76/3,29/2,-4/3,1/12])
    g_flip = np.flip(g)
    #######  end of precomputed items ##########

    #set up right hand side of tria dianganl system equation
    Y=np.zeros_like(f)
    # compute L.H.S of equation
    Y[0] = np.dot(f[0:5],g)
    Y[N-1] = np.dot(f[-5:N],g_flip)
    Y[1:N-1] = ((f[0:N-2]+f[2:N])*a-2*(c+b+a)*f[1:N-1])
    Y[3:N-3] += ((f[1:N-5]+f[5:N-1])*b+(f[0:N-6]+f[6:N])*c)
    Y[1] += ((f[-2]+f[3])*b+(f[-3]+f[4])*c)
    Y[2] += ((f[0]+f[4])*b+(f[-2]+f[5])*c)
    Y[N-3] += (f[N-5]+f[N-1])*b+(f[N-6]+f[1])*c
    Y[N-2] += (f[N-4]+f[1])*b+(f[N-5]+f[2])*c

    #solves banded system of equations
    d2f = scipy.linalg.solve_banded((1, 1), ab, Y/(h**2))
    return d2f

def analyzefd():
    """
    Question 2.1 ii)
    Add input/output as needed

    """
    #for graph1
    #set N and define f and exact solution f''
    N=200
    x=np.arange(N+1)/N
    h=x[1]-x[0]
    f=3*np.cos(8*np.pi*x)/200 - 2*np.sin(20*np.pi*x)/200
    fdiv=((20*np.pi)**2)*2*np.sin(20*np.pi*x)/200-3*((8*np.pi)**2)*np.cos(8*np.pi*x)/200
    #compute F.D approx
    w=newdiff(f,h)
    d2f=np.zeros(N+1)
    d2f[1:-1] = (f[2:]-2*f[1:-1]+f[:-2])/(h**2)
    err1=np.abs(fdiv-w)/np.abs(fdiv)
    err2=np.abs(fdiv-d2f)/np.abs(fdiv)
    #plot graph
    fig, ax1 = plt.subplots(figsize =(12,8))
    ax1.plot(x,err1,label="Compact")
    ax1.plot(x,err2,label="Centered")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Euclidean % error of approximations')
    ax1.set_ylim((-0.001,0.1))

    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(x,fdiv,label="Exact solution of f''",color='y',linestyle="--")
    ax2.plot(x,f*500,label="Exact solution of 500f ",color='g',linestyle="--")
    ax2.set_ylabel('Exact Solutions')
    ax2.tick_params(axis='y')
    ax2.set_ylim((-70,100))

    ax1.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.legend(loc="upper center", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('Comparing performance Centered and Compact F.D on periodic function f(x), N=200', y=1.01, fontsize=17)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()

    #range of kh values to vary
    kh=np.arange(0,400)/100
    kh_sq=np.power(kh,2)

    #compute approx of centered
    w1=2*(1-np.cos(kh))

    #set coefficents as in newdiff
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140

    b=b/4
    c=c/9

    #compute approx of compact
    wave2=2*((1-np.cos(kh))*a+b*(1-np.cos(2*kh))+c*(1-np.cos(3*kh)))/(1+2*alpha*np.cos(kh))

    #plot graph 3
    plt.figure()
    plt.plot(kh,kh_sq,label='Exact')
    plt.plot(kh,w1,label='2nd Order Centered')
    plt.plot(kh,wave2,label='Compact')
    plt.xlabel("hk")
    plt.ylabel(r"$(kh')^2$")
    plt.title("Wavenumber analysis of finite difference approximations")
    plt.legend()
    plt.show()

    #plot graph4
    plt.figure()
    plt.plot(kh,(kh_sq-w1)/kh_sq,label='2nd Order Centered')
    plt.plot(kh,(kh_sq-wave2)/kh_sq,label='Compact')
    plt.plot(kh,kh*0+0.01,label='1% error')
    plt.plot(kh,kh*0+0.1,label='10% error')
    plt.xlabel("hk")
    plt.ylabel("% error w.r.t $(kh')^2$ ")
    plt.title("Error of finite difference methods for changing wavenumber")
    plt.legend()
    plt.show()



    return None #modify as needed


def dynamics():
    """
    Question 2.2
    Add input/output as needed

    """
    #plots figure of f anf g contors
    phi=0.3
    kappa=1.7
    mu=0.4*kappa
    f1,g1=microbes(phi,kappa,mu,L = 1024,Nx=1024,Nt=48001,T=96000,display=True)

    #generates data to compare changing kappa
    kappa1=1.5
    mu1=0.4*kappa1
    kappa2=1.7
    mu2=0.4*kappa2
    kappa3=2.0
    mu3=0.4*kappa3
    Nx=1024
    L=1024
    f1,g1,t1,x1=microbes(phi,kappa1,mu1, L,Nx,Nt=150001,T=96000,display=False,extraoutputs=True)
    f2,g2,t2,x2=microbes(phi,kappa2,mu2, L,Nx,Nt=150001,T=96000,display=False,extraoutputs=True)
    f3,g3,t3,x3=microbes(phi,kappa3,mu3, L,Nx,Nt=150001,T=96000,display=False,extraoutputs=True)

    #work out means and var of a large time frame
    dt=1000
    dt2=4
    mean1=np.mean(f1[-dt:-1:dt2,:],axis=1)
    var1=np.var(f1[-dt:-1:dt2,:],axis=1)
    mean2=np.mean(f2[-dt:-1:dt2,:],axis=1)
    var2=np.var(f2[-dt:-1:dt2,:],axis=1)
    mean3=np.mean(f3[-dt:-1:dt2,:],axis=1)
    var3=np.var(f3[-dt:-1:dt2,:],axis=1)

    #plot of mean
    fig, ax1 = plt.subplots(figsize =(14,10))

    ax1.plot(t1[-dt:-1:dt2],mean1,label=r"mean Kappa=1.5",linestyle="--")
    ax1.plot(t1[-dt:-1:dt2],mean2,label=r"mean Kappa=1.7",linestyle="--")
    ax1.plot(t1[-dt:-1:dt2],mean3,label=r"mean Kappa=2.0",linestyle="--")

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean')

    ax1.tick_params(axis='y')
    ax1.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title(r'How mean of f(x) changes for large time for varying $\kappa$  ', fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()
    fig, ax1 = plt.subplots(figsize =(14,10))
    #plot of variance
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variance')

    ax1.tick_params(axis='y')


    ax1.plot(t1[-dt:-1:dt2],var1,label=r"$\sigma^2$ Kappa=1.5",linestyle="--")
    ax1.plot(t1[-dt:-1:dt2],var2,label=r"$\sigma^2$ Kappa=1.7",linestyle="--")
    ax1.plot(t1[-dt:-1:dt2],var3,label=r"$\sigma^2$ Kappa=2.0",linestyle="--")
    ax1.set_ylabel('Variance')


    ax1.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title(r'How variance of f(x) changes for large time for varying $\kappa$ ', fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()

    #calculating means with respect to time
    dt=80000
    dt2=1
    mean1=np.mean(f1[-dt:-1:dt2,:],axis=0)
    var1=np.var(f1[-dt:-1:dt2,:],axis=0)
    mean2=np.mean(f2[-dt:-1:dt2,:],axis=0)
    var2=np.var(f2[-dt:-1:dt2,:],axis=0)
    mean3=np.mean(f3[-dt:-1:dt2,:],axis=0)
    var3=np.var(f3[-dt:-1:dt2,:],axis=0)
    #plots means and varaince
    fig, ax1 = plt.subplots(figsize =(14,10))
    ax1.plot(x1,mean1,label=r"$\mu$ Kappa=1.5",linestyle="--")
    ax1.plot(x2,mean2,label=r"$\mu$ Kappa=1.7",linestyle="--")
    ax1.plot(x3,mean3,label=r"$\mu$ Kappa=2.0",linestyle="--")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Mean')

    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    ax2.plot(x1,var1,label=r"$\sigma^2$ Kappa=1.5")
    ax2.plot(x2,var2,label=r"$\sigma^2$ Kappa=1.7")
    ax2.plot(x3,var3,label=r"$\sigma^2$ Kappa=2.0")
    ax2.set_ylabel('Variance')
    ax2.tick_params(axis='y')

    ax1.legend(loc="lower center", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.legend(loc="lower left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('How Mean and variance for f(x) over x for different kappa, over 80000 time steps', fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()

    #generate data for smaller time fram
    D = scipy.spatial.distance.pdist(f1[-2000:-10,:])
    k=18
    n=28
    m_scale=len(f1[0,:])*(len(f1[0,:])-1)/2
    eps=np.logspace(5, 28, base=1.1)
    C=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C[i] = (D1.size)*m_scale
    y=np.polyfit(np.log(eps)[-n:-k],np.log(C)[-n:-k],1)

    D = scipy.spatial.distance.pdist(f2[-2000:-10,:])
    eps=np.logspace(5, 28, base=1.1)
    C2=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C2[i] = (D1.size)*m_scale
    y2=np.polyfit(np.log(eps)[-n:-k],np.log(C2)[-n:-k],1)

    D = scipy.spatial.distance.pdist(f3[-2000:-10,:])
    eps=np.logspace(5, 28, base=1.1)
    C3=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C3[i] = (D1.size)*m_scale
    y3=np.polyfit(np.log(eps)[-n:-k],np.log(C3)[-n:-k],1)



    plt.figure(figsize =(12,10))
    plt.title("Log plot for Correlation Sum vs Eplislon ")
    plt.scatter(np.log(eps)[-55:-1],np.log(C)[-55:-1],label="kappa=1.5,C(eplison)")
    plt.plot(np.log(eps)[-50:-4],y[0]*np.log(eps)[-50:-4]+y[1],label="kappa=1.5,regression, slope=%.2f"%y[0])
    plt.scatter(np.log(eps)[-55:-1],np.log(C2)[-55:-1],label="kappa=1.7,C(eplison)",color='r')
    plt.plot(np.log(eps)[-50:-4],y2[0]*np.log(eps)[-50:-4]+y2[1],label="kappa=1.7,regression, slope=%.2f"%y2[0],color='r')
    plt.scatter(np.log(eps)[-55:-1],np.log(C3)[-55:-1],label="kappa=2.0,C(eplison)",color='g')
    plt.plot(np.log(eps)[-50:-4],y3[0]*np.log(eps)[-50:-4]+y3[1],label="kappa=2.0,regression, slope=%.2f"%y3[0],color='g')
    plt.legend()
    plt.ylabel("log(C(eplison)")
    plt.xlabel("log(eplison)")
    plt.show()

    #set basic variables and work out correlation sum for different kappa
    D = scipy.spatial.distance.pdist(f1[-20000:-10,:])
    k=18
    n=28
    m_scale=len(f1[0,:])*(len(f1[0,:])-1)/2
    eps=np.logspace(5, 28, base=1.1)
    C=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C[i] = (D1.size)*m_scale
    y=np.polyfit(np.log(eps)[-n:-k],np.log(C)[-n:-k],1)

    #kapp1.5
    D = scipy.spatial.distance.pdist(f2[-20000:-10,:])
    eps=np.logspace(5, 28, base=1.1)
    C2=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C2[i] = (D1.size)*m_scale
    y2=np.polyfit(np.log(eps)[-n:-k],np.log(C2)[-n:-k],1)

    A = np.load('data3.npy')
    D = scipy.spatial.distance.pdist(f3[-20000:-10,:])
    eps=np.logspace(5, 28, base=1.1)
    C3=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C3[i] = (D1.size)*m_scale
    y3=np.polyfit(np.log(eps)[-n:-k],np.log(C3)[-n:-k],1)


    #plot graph
    plt.figure(figsize =(12,10))
    plt.title("Log plot for Correlation Sum vs Eplislon ")
    plt.scatter(np.log(eps)[-55:-1],np.log(C)[-55:-1],label="kappa=1.5,C(eplison)")
    plt.plot(np.log(eps)[-50:-4],y[0]*np.log(eps)[-50:-4]+y[1],label="kappa=1.5,regression, slope=%.2f"%y[0])
    plt.scatter(np.log(eps)[-55:-1],np.log(C2)[-55:-1],label="kappa=1.7,C(eplison)",color='r')
    plt.plot(np.log(eps)[-50:-4],y2[0]*np.log(eps)[-50:-4]+y2[1],label="kappa=1.7,regression, slope=%.2f"%y2[0],color='r')
    plt.scatter(np.log(eps)[-55:-1],np.log(C3)[-55:-1],label="kappa=2.0,C(eplison)",color='g')
    plt.plot(np.log(eps)[-50:-4],y3[0]*np.log(eps)[-50:-4]+y3[1],label="kappa=2.0,regression, slope=%.2f"%y3[0],color='g')
    plt.legend()
    plt.ylabel("log(C(eplison)")
    plt.xlabel("log(eplison)")
    plt.show()

    D = scipy.spatial.distance.pdist(data3[:,36,:])
    k=30
    n=50
    m_scale=len(data3[:,0,0])*(len(data3[0,:,0])-1)/2
    eps=np.logspace(5, 28, base=1.5)
    C=np.zeros_like(eps)

    data3 = np.load('data3.npy')
    data3= dat3[:,:,20:-1]
    D = scipy.spatial.distance.pdist(data3[:,108,:])
    eps=np.logspace(5, 28, base=1.1)
    C2=np.zeros_like(eps)
    for i in range(len(eps)):
      D1 = D.copy()
      D1 = D1[D1<eps[i]]
      C2[i] = (D1.size)*m_scale
    y2=np.polyfit(np.log(eps)[-n:-k],np.log(C2)[-n:-k],1)

    plt.figure(figsize =(12,10))
    plt.title("Log plot for Correlation Sum vs Eplislon, for Theta=3\pi/4 ")
    plt.scatter(np.log(eps)[-55:-1],np.log(C2)[-55:-1],label="kappa=1.7,C(eplison)",color='r')
    plt.plot(np.log(eps)[-50:-4],y2[0]*np.log(eps)[-50:-4]+y2[1],label="kappa=1.7,regression, slope=%.2f"%y2[0],color='r')
    plt.legend()
    plt.ylabel("log(C(eplison)")
    plt.xlabel("log(eplison)")
    plt.show()
    return None #modify as needed

if __name__=='__main__':
    x=None
    #Add code here to call functions above and
    #generate figures you are submitting
