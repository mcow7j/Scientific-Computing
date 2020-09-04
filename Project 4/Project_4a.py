"""Scientific Computation Project 3, part 1
Your CID here 01059624
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import scipy

def hfield(r,th,h,levels=50):
    """Displays height field stored in 2D array, h,
    using polar grid data stored in 1D arrays r and th.
    Modify as needed.
    """
    thg,rg = np.meshgrid(th,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,h,levels)
    plt.axis('equal')
    return None

def repair1(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data


    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))
            mlist[i].append(j)
            nlist[j].append(i)

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for k in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0

                    for j in mlist[m]:
                        Bfac += B[n,j]**2
                        Rsum = 0
                        for k in range(p):
                            if k != n: Rsum += A[m,k]*B[k,j]
                        Asum += (R[m,j] - Rsum)*B[n,j]

                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                    #Add code here to update B[m,n]
                    Afac = 0.0
                    Bsum = 0

                    for i in nlist[n]:
                        Afac += A[i,m]**2
                        Rsum = 0
                        for k in range(p):
                            if k != m: Rsum += B[k,n]*A[i,k]
                        Bsum += (R[i,n] - Rsum)*A[i,m]
                    B[m,n] = Bsum/(Afac+l)

        dA[k] = np.sum(np.abs(A-Aold))
        dB[k] = np.sum(np.abs(B-Bold))
        if dA[k]+dB[k]<=10**(-12):
          print("converged")
    return A,B


def repair2(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R. Efficient and complete version of repair1.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    a,b = R.shape
    #create index matrix
    index_mat= R!= -1000

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for k in range(niter):
        Aold = A.copy()
        Bold = B.copy()
        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac=np.dot(B[n,index_mat[m,:]],B[n,index_mat[m,:]])
                    Rsum=np.dot(A[m,:p],B[:p,index_mat[m,:]])-A[m,n]*B[n,index_mat[m,:]]
                    Asum= ((R[m,index_mat[m,:]] - Rsum)*B[n,index_mat[m,:]]).sum()
                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                    Afac=np.dot(A[index_mat[:,n],m],A[index_mat[:,n],m])
                    Rsum=np.dot(A[index_mat[:,n],:p],B[:p,n])-B[m,n]*A[index_mat[:,n],m]
                    Bsum= ((R[index_mat[:,n],n] - Rsum)*A[index_mat[:,n],m]).sum()
                    B[m,n] = Bsum/(Afac+l)

        dA[k] = np.sum(np.abs(A-Aold))
        dB[k] = np.sum(np.abs(B-Bold))
        if dA[k]+dB[k]<=10**(-12) and k>2:
          print("converged")
          return A,B
    return A,B


def outwave(r0):
    """
    Question 1.2i)
    Calculate outgoing wave solution at r=r0
    See code/comments below for futher details
        Input: r0, location at which to compute solution
        Output: B, wave equation solution at r=r0

    """
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')
    T=np.arange(0,20)*np.pi/80

    Nr,Ntheta,Nt = A.shape
    B = np.zeros((Ntheta,Nt),dtype=complex)

    #find 2d fourier transfrom
    c=np.fft.fft2(data2[0,:,:])

    #set up frequences and indexs
    mind=[]
    nind=[]
    #freq_m=np.zeroslike(c[0,:])
    #freq_n=np.zeroslike(c[:,0])

    #frequences here, not correct
    freq_m=np.fft.ifftshift(T)
    freq_n=np.fft.ifftshift(th)

    #go through c looking for significant coefficents
    for m in range(Nt):
        for n in range(Ntheta):
            if np.abs(c[n,m])>0.00001:
                nind.append(n)
                mind.append(m)

    #sum of frequences
    for n,m in zip(nind,mind):
        B+=scipy.special.hankel1(freq_n[n], freq_m[m]*r0)*c[n,m]* \
        np.einsum("k,t->kt",np.exp(-1j*freq_n[n]*th),np.exp(-1j*freq_m[m]*T)) \

    return np.real(B)

def analyze1():
    """
    Question 1.2ii)
    Add input/output as needed

    """
    #import data3
    data3 = np.load('data3.npy')
    #genrate r ,t,th
    t=np.arange(len(data3[0,0,:]))
    r=np.linspace(1,5,data3.shape[0])
    th=np.linspace(0,2*np.pi,data3.shape[1])

    #3d plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    rr, tt = np.meshgrid(r, t, indexing='ij')
    # Plot the surface
    ax.plot_surface(rr,tt,data3[:,36,:],cmap=plt.cm.plasma,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('R')
    ax.set_ylabel('Time')
    ax.set_zlabel('H')
    plt.title('Theta=$\pi$/4', fontsize=20)
    plt.show()

    #contour plots
    rr, tt = np.meshgrid(r, t, indexing='ij')

    plt.figure(figsize=(12, 12))
    plt.contourf(tt,rr,data3[:,36,:],200,cmap=plt.cm.plasma)
    plt.xlabel("time")
    plt.ylabel("R")
    plt.clim(-6,5)
    plt.colorbar()
    plt.title('Theta=$\pi$/4', fontsize=20)
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.contourf(tt,rr,data3[:,108,:],200,cmap=plt.cm.plasma)
    plt.xlabel("time")
    plt.ylabel("R")
    plt.clim(-6,5)
    plt.colorbar()
    plt.title('Theta=3$\pi$/4', fontsize=20)
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.contourf(tt,rr,data3[:,180,:],200,cmap=plt.cm.plasma)
    plt.xlabel("time")
    plt.ylabel("R")
    plt.clim(-6,5)
    plt.colorbar()
    plt.title('Theta=5$\pi$/4', fontsize=20)
    plt.show()

    #generate mean and varaince dat
    mean1=np.mean(data3[:,36,:],axis=0)
    var1=np.var(data3[:,36,:],axis=0)
    mean2=np.mean(data3[:,108,:],axis=0)
    var2=np.var(data3[:,108,:],axis=0)
    mean3=np.mean(data3[:,180,:],axis=0)
    var3=np.var(data3[:,180,:],axis=0)

    #plot graph
    fig, ax1 = plt.subplots(figsize =(12,8))

    ax1.plot(t,mean1,label=r"mean $\theta=\pi$/4")
    ax1.plot(t,mean2,label=r"mean $\theta=\pi$/4")
    ax1.plot(t,mean3,label=r"mean $\theta=5 \pi$/4")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(t,var1,label=r"var $\theta=\pi$/4",linestyle="--")
    ax2.plot(t,var2,label=r"var $\theta=3\pi$/4",linestyle="--")
    ax2.plot(t,var3,label=r"var $\theta=5\pi$/4",linestyle="--")
    ax2.set_ylabel('Variance')
    ax2.tick_params(axis='y')

    ax1.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.legend(loc="upper right", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('Mean and variance of h for 1<r<5 and fixed values of theta ', fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()

    #figure 7
    mean1=np.mean(data3[:,36,:],axis=1)
    var1=np.var(data3[:,36,:],axis=1)
    mean2=np.mean(data3[:,108,:],axis=1)
    var2=np.var(data3[:,108,:],axis=1)
    mean3=np.mean(data3[:,180,:],axis=1)
    var3=np.var(data3[:,180,:],axis=1)

    fig, ax1 = plt.subplots(figsize =(12,8))

    ax1.plot(r,mean1,label=r"mean $\theta=\pi$/4")
    ax1.plot(r,mean2,label=r"mean $\theta=3\pi$/4")
    ax1.plot(r,mean3,label=r"mean $\theta=5 \pi$/4")
    #ax1.plot(mem_sav2,forb_array2,label="Frobenius error time")
    ax1.set_xlabel('r')
    ax1.set_ylabel('Mean')

    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    ax2.plot(r,var1,label=r"var $\theta=\pi$/4",linestyle="--")
    ax2.plot(r,var2,label=r"var $\theta=3\pi$/4",linestyle="--")
    ax2.plot(r,var3,label=r"var $\theta=5\pi$/4",linestyle="--")
    #ax2.plot(mem_sav2,error_array2,label="average error time",color='m')
    ax2.set_ylabel('Variance')
    ax2.tick_params(axis='y')

    ax1.legend(loc="upper left", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.legend(loc="upper right", fontsize=12,fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.title('Mean and variance of h for 0<t<20 and fixed values of theta ', fontsize=20)
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plt.grid(b=None)
    plt.show()
    return None #modify as needed




def reduce(H,inputs=(12,80,12)):
    """
    Question 1.3: Construct one or more arrays from H
    that can be used by reconstruct
    Input:
        H: 3-D data array
        inputs: k,k_u,k_v which dictacte the quality of SVD decompostions of H (the rank)
    Output:
        arrays: a tuple containing the arrays produced from H
    """
    #1 corresponds to of SVD U_red,2 to SVD of V_red
    k,k1,k2 = inputs

    #SVD of data 3
    U,S,Vh = np.linalg.svd(H)

    U_red = U[:,:,0:k]
    Vh_red = Vh[:,0:k,:]
    S_red = S[:,0:k]
    #set up Final SVD decompostions of U_red and Vh_red
    U_red1 = np.zeros((k,300,k1))
    Vh_red1 = np.zeros((k,k1,289))
    S_red1 = np.zeros((k,k1))
    U_red2 = np.zeros((k,300,k2))
    Vh_red2 = np.zeros((k,k2,119))
    S_red2 = np.zeros((k,k2))
    #iterate over slices of first plane
    for j in range(k):
        #SVD of U_red
        U1,S1,Vh1 = np.linalg.svd(U_red[:,:,j])
        U_red1[j,:,0:k1] = U1[:,0:k1]
        Vh_red1[j,0:k1,:] = Vh1[0:k1,:]
        S_red1[j,0:k1] = S1[0:k1]
        #SVD of VH_red
        U2,S2,Vh2 = np.linalg.svd(Vh_red[:,j,:])
        U_red2[j,:,0:k2] = U2[:,0:k2]
        Vh_red2[j,0:k2,:] = Vh2[0:k2,:]
        S_red2[j,0:k2] = S2[0:k2]

    #Add code here
    arrays = (U_red1,Vh_red1,S_red1,U_red2,Vh_red2,S_red2,S_red)
    return arrays


def reconstruct(arrays,inputs=()):
    """
    Question 1.3: Generate matrix with same shape as H (see reduce above)
    that has some meaningful correspondence to H
    Input:
        arrays: tuple generated by reduce
        inputs: can be used to provide other input as needed
    Output:
        Hnew: a numpy array with the same shape as H
    """
    #extract arrays from input
    U_red1,Vh_red1,S_red1,U_red2,Vh_red2,S_red2,S_red = arrays
    #reconstruct U
    U_red = np.einsum("jmk,jkn,jk->mnj",U_red1,Vh_red1,S_red1)
    #reconstruct V
    V_red = np.einsum("jmk,jkn,jk->mjn",U_red2,Vh_red2,S_red2)
    #reconstruct data3
    Hnew = np.einsum("imk,ikn,ik->imn",U_red,V_red,S_red)
    return Hnew


if __name__=='__main__':
    x=None
    #Add code here to call functions above and
    #generate figures you are submitting
