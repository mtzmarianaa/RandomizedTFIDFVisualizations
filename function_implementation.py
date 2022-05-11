# FUNCTIONS PRESENTED IN DRINEAS, KANNAN & MAHONEY PAPERS
from cProfile import label
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from random import randint, random
from math import log, floor
from scipy.linalg import hadamard, qr
import pandas as pd
import time
import matplotlib.animation as animation


######### Define nice colormaps
colormap1 = plt.cm.get_cmap('viridis')
sm1 = plt.cm.ScalarMappable(cmap=colormap1)
colormap2 = plt.cm.get_cmap('magma')
sm2 = plt.cm.ScalarMappable(cmap=colormap2)
colormap3 = plt.cm.get_cmap('plasma')
sm3 = plt.cm.ScalarMappable(cmap=colormap3)
############
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        t = str(time.time() - startTime_for_tictoc)
        #print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        t = ''
    return t

#### Functions from papers 

def basicMatrixMultiplications(A, B, c, p):
    '''
    Algorithm proposed on Drineas, Kannan & Mahoney's first paper (out of 3), p. 138
    IN    - A : m x n real matrix
          - B : n x q real matrix
          - c : real number such that 1<= c <= n
          - p : 1xn vector with initial probability distribution
    OUT   - C : m x c real matrix
          - R : c x p real matrix such that AB approx CR
    '''
    m, n = np.shape(A)
    q = np.shape(B)[1]
    possible_indices = range( n )
    C = np.zeros((m, c))
    R = np.zeros((c, q))
    for t in range(c):
        it = np.random.choice( possible_indices, size=1, p= p  )
        C[:, t] = A[:, it[0]]/sqrt(c*p[it[0]])
        rr = B[it[0], :]/sqrt(c*p[it[0]])
        R[t, :] = B[it[0], :]/sqrt(c*p[it[0]])
    return C, R

def optimal_probabilities(A, B):
    '''
    Proposed optimal initial probability as on Drines, Kannan & Mahoney's first paper (p. 141)
    IN    - A : m x n real matrix
          - B : n x q real matrix
    OUT   - p : 1xn vector with initial optimal probability distribution    
    '''
    n = len(B)
    p = [0]*n
    normalizing_sum = 0
    for i in range(n):
        normalizing_sum = normalizing_sum + norm(A[:, i])* norm(B[i, :]) 
        p[i] = norm(A[:, i])* norm(B[i, :]) 
    p = p/normalizing_sum
    return p

def plot_withTheoreticalBounds(A, B, cs, error_f, error_s, p, beta, logPlot = True, Ylimits = False):
    '''
    Method that plots nicely graphs with the errors (Frobenius + spectral) of AB vs CR (from the Basic Matrix Multiplication
    method). All the theoretical bounds that are presented here follow up from the Drines, Kannan & Mahoney's first paper.
    IN    - A : m x n real matrix
          - B : n x q real matrix
          - error_f : list with squared Frobenius errors after simulating with basicMatrixMultiplications
          - error_s :list with saquared spectral errors after simulating with basicMatrixMultiplications
          - p : 1xn vector with initial optimal probability distribution  
          - beta : see p. 142
    '''
    m, n = np.shape(A)
    q = np.shape(B)[1]
    # Expectation and from lemma 4, p 140-141 IF beta = 1
    n_c = len(cs)
    if beta == 1:
        expectation_f = []
        for i in range(n_c):
            sum = 0
            for j in range(n):
                sum = sum + (norm(A[:, j])**2* norm(B[j, :])**2)/(cs[i]*p[j]) 
            expectation_f = expectation_f + [ sum - (1/cs[i])*norm(np.matmul(A,B), 'fro')  ]
    # Expectation bound from theorem 1, p. 142
    nA = norm(A, 'fro')**2
    nB = norm(B, 'fro')**2
    expect_bound_f = []
    error_bound_f_p10 = [] # meaning delta = 0.90
    n1 = (1 + sqrt( (8/beta)*log(1/0.90) ))**2
    error_bound_f_p20 = [] # meaning delta = 0.80
    n2 = (1 + sqrt( (8/beta)*log(1/0.80) ))**2
    error_bound_f_p30 = [] # meaning delta = 0.70
    n3 = (1 + sqrt( (8/beta)*log(1/0.70) ))**2
    error_bound_f_p40 = [] # meaning delta = 0.60
    n4 = (1 + sqrt( (8/beta)*log(1/0.60) ))**2
    error_bound_f_p50 = [] # meaning delta = 0.50
    n5 = (1 + sqrt( (8/beta)*log(1/0.50) ))**2
    error_bound_f_p60 = [] # meaning delta = 0.40
    n6 = (1 + sqrt( (8/beta)*log(1/0.40) ))**2
    error_bound_f_p70 = [] # meaning delta = 0.30
    n7 = (1 + sqrt( (8/beta)*log(1/0.30) ))**2
    error_bound_f_p80 = [] # meaning delta = 0.20
    n8 = (1 + sqrt( (8/beta)*log(1/0.20) ))**2
    error_bound_f_p90 = [] # meaning delta = 0.10
    n9 = (1 + sqrt( (8/beta)*log(1/0.10) ))**2
    for i in range(n_c):
        expect_bound_f = expect_bound_f + [1/(beta*cs[i])*nA*nB]
        error_bound_f_p10 = error_bound_f_p10 + [n1/(beta*cs[i])*nA*nB]
        error_bound_f_p20 = error_bound_f_p20 + [n2/(beta*cs[i])*nA*nB]
        error_bound_f_p30 = error_bound_f_p30 + [n3/(beta*cs[i])*nA*nB]
        error_bound_f_p40 = error_bound_f_p40 + [n4/(beta*cs[i])*nA*nB]
        error_bound_f_p50 = error_bound_f_p50 + [n5/(beta*cs[i])*nA*nB]
        error_bound_f_p60 = error_bound_f_p60 + [n6/(beta*cs[i])*nA*nB]
        error_bound_f_p70 = error_bound_f_p70 + [n7/(beta*cs[i])*nA*nB]
        error_bound_f_p80 = error_bound_f_p80 + [n8/(beta*cs[i])*nA*nB]
        error_bound_f_p90 = error_bound_f_p90 + [n9/(beta*cs[i])*nA*nB]
    
    # Plot for the Frobenius errors
    plt.figure()
    plt.scatter(cs, error_f, color = colormap2(70), alpha=0.3, marker=".", label = "Error")
    if beta == 1:
        plt.plot(cs, expectation_f, color = colormap2(200), linewidth = 2, label = "Expected error")
    plt.plot(cs, expect_bound_f, color = colormap2(200), linewidth = 1, alpha = 0.7, linestyle = '-', label = "Bound on expected error")
    plt.plot(cs, error_bound_f_p10, color = colormap2(170), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.10")
    plt.plot(cs, error_bound_f_p20, color = colormap2(150), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.20")
    plt.plot(cs, error_bound_f_p30, color = colormap2(130), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.30")
    plt.plot(cs, error_bound_f_p40, color = colormap2(110), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.40")
    plt.plot(cs, error_bound_f_p50, color = colormap2(90), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.50")
    plt.plot(cs, error_bound_f_p60, color = colormap2(70), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.60")
    plt.plot(cs, error_bound_f_p70, color = colormap2(50), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.70")
    plt.plot(cs, error_bound_f_p80, color = colormap2(30), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.80")
    plt.plot(cs, error_bound_f_p90, color = colormap2(10), linewidth = 1, alpha = 0.5, linestyle = ':', label = "Bound error with probability 0.90")
    if logPlot:
        plt.yscale('log')
    if Ylimits != False:
        plt.ylim(Ylimits)
    plt.xlabel('c')
    plt.ylabel('Frobenius error^2')
    plt.show(block=False)
    plt.title("Squared Frobenius error for different sizes of approximations, " + str(m) + "x" + str(n) + " and " + str(n)+ "x" + str(q) )
    plt.legend(framealpha=1, frameon=True)
    
    # Plot for the Spectral errors
    plt.figure()
    plt.scatter(cs, error_s, color = colormap2(80), alpha=0.3, marker=".", label = "Error")
    plt.xlabel('c')
    plt.ylabel('Spectral error^2')
    if logPlot:
        plt.yscale('log')
    if Ylimits != False:
        plt.ylim(Ylimits)
    plt.show(block=False)
    plt.title("Squared spectral error for different sizes of approximations, " + str(m) + "x" + str(n) + " and " + str(n)+ "x" + str(q) )
    plt.legend(framealpha=1, frameon=True)
    
def randomizedProjectorRST(A, l, i, k = 3):
    '''
    From Rokhlin, Szlam, Tygert paper A Randomized Algorithm For Principal Component Analysis, algorithm 4.1
    IN :   
           A       : mxn matrix to be factorized
           l       : paramter for the size of G, the Gaussian random matrix lxm
           i       : parameter for exponentiation
           k       : order of approximation wanted
    OUT :
           U       : approximated k top left singular vectors mxk
           Sigma   : appeoximated k top singular values kxk
           V       : approximated k top right singular vectors kxn
    '''
    m = A.shape[0]
    # STEP 1
    # First step, generate a real lxm matrix G whose entries are iid Gaussian rv with zero mean and unit variance
    G = np.random.normal(loc= 0.0, scale = 1.0, size = [l, m])
    # Then compute the lxn produc matrix
    R = G@np.linalg.matrix_power(A@np.transpose(A), i)@A
    # STEP 2
    # Using an SVD form a real nxk matrix Q whose columns are orthonormal, such that there exists a real kxl matrix S 
    _, _, vtHat = np.linalg.svd(R, full_matrices = False)
    Q = np.transpose(vtHat[1:k, :])
    # STEP 3
    # Compute the mxk product matrix
    T = A@Q
    # STEP 4
    # Form an SVD of T
    U, Sigma, W =  np.linalg.svd(T, full_matrices = False)
    # STEP 5
    # Compute the n x k product matrix
    V = Q@np.transpose(W)
    return U, Sigma, V

def SVD_Blanczos(A, l, i, Sigma_full = False):
    '''
    From Rokhlin, Szlam, Tygert paper A Randomized Algorithm For Principal Component Analysis, algorithm 4.4 called Blanczos
    IN :   
           A          : mxn matrix to be factorized
           l          : paramter for our approximated matrix C of size mxc
           i          : order of approximation wanted
           Sigma_full : transition probabilities
    OUT :
           U          : approximated left singular vectors
           Sigma      : appeoximated singular values
           V          : approximated right singular vectors
    ''' 
    m = A.shape[0]
    n = A.shape[1]
    # STEP 1
    # Using a random number generator form a real lxm matrix G whose entries are iid Gaussian and compute the lxn matrices
    G = np.random.normal(loc= 0.0, scale = 1.0, size = [l, m])
    R = np.zeros(( (i+1)*l, n ))
    R_temp = G@A
    R[0:(l), :] = R_temp
    for j in range(i):
        R_temp = R_temp@np.transpose(A)@A
        R[ (j+1)*l:(j+2)*(l), : ] = R_temp
    # STEP 2
    # Using QR decomposition form a real n x ((i+1)l) matrix Q whose columns are orthonormal
    Q, S = qr(np.transpose(R))
    # STEP 3
    # Compute the m x ( (i+1)l ) product matrix
    T = A@Q
    # STEP 4
    # Form an SVD of T
    U, Sigma, Wt = np.linalg.svd(T)
    # STEP 5
    # Compute the n x( (i+1)l ) product matrix
    V = Q@np.transpose(Wt)
    if Sigma_full:
        S_t = Sigma
        Sigma = np.zeros((m,n))
        np.fill_diagonal(Sigma, S_t)
    return U, Sigma, V

def linearTimesSVD(A, c, k, p):
    '''
    LinearTimesSVD algorithm found in Drineas, Kannan, Mahoney paper 'Fast Monte Carlo Algorithms for Matrices II'
    Here 1 <= k <= c <=n      p is vector with transition probabilities
    IN :   
           A      : mxn matrix to be factorized
           c      : paramter for our approximated matrix C of size mxc
           k      : order of approximation wanted
           p      : transition probabilities
    OUT :
           H      : approximated k top left singular vectors
           SigmaC : appeoximated k top singular values
           Yt     : approximated k top right singular vectors
    '''
    m, n = np.shape(A)
    C = np.zeros((m, c))
    possible_indices = range( n )
    for t in range(c):
        it = np.random.choice( possible_indices, size=1, p= p  )
        C[:, t] = A[:, it[0]]/sqrt(c*p[it[0]])
    CCt = np.transpose(C)@C
    Y, SigmaC, _ = np.linalg.svd(CCt)
    H = np.zeros((m, k))
    for t in range(k):
        H[:, t] = C.dot(Y[:, t])
    H = H/SigmaC[0:k]
    return H, SigmaC[0:k], C

def plotMethodsErrorsvsExact(A, c, i, num_k = 25):
    '''
    Function that implements all three methods: SVD_Blanczcos, LinearTimesSVD
    and also calculates the exact SVD and compares the approximations
    '''
    m = A.shape[0]
    n = A.shape[1]
    U, S, Vt = np.linalg.svd(A) # EXACT SVD
    p = optimal_probabilities(A, np.transpose(A)) # We're going to use to optimal transition probabilities
    kTemp = floor(min(m, n)/4)
    l = 2*kTemp
    UBlanczos, SBlanczos, VtBlanczos = SVD_Blanczos(A, l, i)
    VtBlanczos = np.transpose(VtBlanczos)
    kVec = np.linspace(2, kTemp, num = num_k)
    kVec = [int(k) for k in set(kVec)]
    error_exact = []
    error_linTime = []
    error_Blanczos = []
    for k in kVec:
        error_exact.append( norm( U[:, range(k)]@np.diag( S[range(k)] )@Vt[range(k), :] - A  , ord = 2 ) )
        Hlin, _, _ = linearTimesSVD(A, c, k, p)
        error_linTime.append( norm( Hlin@np.transpose(Hlin)@A - A  , ord = 2 ) )
        #UBlanczos[:, range(k)]@np.diag( SBlanczos[range(k)] )@VtBlanczos[range(k), :]
        error_Blanczos.append( norm( UBlanczos[:, range(k)]@np.transpose(UBlanczos[:, range(k)])@A - A  , ord = 2 ) )
    print(error_linTime)
    # Plot such errors
    plt.figure()
    kVec = list(kVec)
    plt.plot( kVec, error_exact, color = colormap2(25), linewidth = 1, label = "Exact SVD")
    plt.plot( kVec, error_linTime, color = colormap2(75), linewidth = 1, label = "Linear time SVD")
    plt.plot( kVec, error_Blanczos, color = colormap2(125), linewidth = 1, label = "Blanczos")
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.title("Error from different methods for low-rank approximations")
    plt.legend(framealpha=1, frameon=True)
    
    plt.figure()
    plt.plot( kVec, error_exact, color = colormap2(25), linewidth = 1)
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.title("Error from exact SVD")
    
    plt.figure()
    plt.plot( kVec, error_linTime, color = colormap2(75), linewidth = 1)
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.title("Error from linear time SVD")
    
    plt.figure()
    plt.plot( kVec, error_Blanczos, color = colormap2(125), linewidth = 1)
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.title("Error from Blanczos")
    
    plt.figure()
    plt.plot( kVec, error_exact, color = colormap2(25), linewidth = 1, label = "Exact SVD")
    plt.plot( kVec, error_linTime, color = colormap2(75), linewidth = 1, label = "Linear time SVD")
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.title("Error from different methods for low-rank approximations")
    plt.legend(framealpha=1, frameon=True)
    
    plt.figure()
    plt.plot( kVec, error_exact, color = colormap2(25), linewidth = 1, label = "Exact SVD")
    plt.plot( kVec, error_Blanczos, color = colormap2(125), linewidth = 1, label = "Blanczos")
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.title("Error from different methods for low-rank approximations")
    plt.legend(framealpha=1, frameon=True)
    
def errorsTable(A):
    '''
    Given a list of values of c, and i to try out (not long lists) computes the errors from linearTimeSVD and Blanczos
    '''
    num_c = 3
    num_k = 3
    i_s = [0, 1, 2]
    m = A.shape[0]
    n = A.shape[1]
    tic()
    U, _, _ = np.linalg.svd(A) # EXACT SVD
    time_svd = toc()
    tic()
    p = optimal_probabilities(A, np.transpose(A)) # We're going to use to optimal transition probabilities
    p_time = toc() # Time taken to calculate the optimal transition probabilities
    kTemp = floor(min(m, n)/4)
    l = 2*kTemp
    kVec = np.linspace(2, kTemp, num = num_k)
    kVec = [int(k) for k in set(kVec)]
    kVec.sort()
    cVec = [int(c) for c in set(np.linspace(l, 3*kTemp, num = num_c))]
    cVec.sort()
    errors_table_lT = pd.DataFrame(columns = ['c', 'k', 'Error', 'Time'] )
    errors_table_B = pd.DataFrame(columns = ['i', 'k', 'Error', 'Time'])
    errors_svd = pd.DataFrame(columns = ['k', 'Error'])
    # We build the SVD error table
    i = 0
    for k in kVec:
        Uapprox = U[:, range(k+1)]
        errors_svd.loc[i] = [ k, norm(A - Uapprox@np.transpose(Uapprox)@A , ord = 2) ]
        i += 1
    # Build the linear Times SVD error table
    i = 0
    for c in cVec:
        for k in kVec:
            tic()
            H, _, _ = linearTimesSVD(A, c, k, p)
            t_l = toc()
            errors_table_lT.loc[i] = [c, k, norm(A - H@np.transpose(H)@A, ord = 2), float(t_l)+float(p_time)]
            i += 1
    # Build the Blanczos error table
    i = 0
    for i_param in i_s:
        tic()
        UBlanczos, _, _ = SVD_Blanczos(A, l, i_param)
        t_b = toc()
        for k in kVec:
            Uapprox = UBlanczos[:, range(k+1)]
            errors_table_B.loc[i] = [ i_param, k, norm( A - Uapprox@np.transpose(Uapprox)@A , ord = 2), t_b ]
            i += 1
    return time_svd, errors_svd, errors_table_lT, errors_table_B

def rotate(angle):
    ax.view_init(azim=angle)
 
def plot_low(Rep, Tags, name_g):
    '''
    Low dimensional visualization given from the Rep matrix (tall matrix), colored dependending on the tags list
    '''
    df = pd.DataFrame(Rep, columns=['C 1', 'C 2', 'C 3'])
    df['Origin'] = Tags
    groupsRep = df.groupby('Origin')
    # 3D REPRESENTATION
    fig1 = plt.figure()
    ax = plt.axes(projection = '3d')
    k = 0
    for name, group in groupsRep:
        ax.scatter( group['C 1'], group['C 2'], group['C 3'], s = 2, label = name, c = colormap3(50*k), alpha = 0.5  )
        k += 1
    plt.legend()
    #rot_animation = animation.FuncAnimation(fig1, rotate, frames=np.arange(0, 362, 2), interval=100)
    #print('Figures/3DRep' + name_g + '.gif')
    #rot_animation.save('Figures/3DRep' + name_g + '.gif', dpi=80, writer='imagemagick')
    plt.title('3D Representation' + name_g)
    plt.show(block = False)
    
    # 2D REPRESENTATION
    k = 0
    fig2 = plt.figure()
    for name, group in groupsRep:
        plt.scatter( group['C 1'], group['C 2'], s = 2, label = name, c = colormap3(50*k), alpha = 0.5 )
        k += 1
    plt.legend()
    plt.title('2D Representation' + name_g)
    plt.show(block=False)
        
        
def LowDimVisualization(A, Tags):
    '''
    Projection onto the singular vectors of a matrix A in order to perform visualizations
    '''
    num_c = 2
    i_s = [0, 1, 2]
    m = A.shape[0]
    n = A.shape[1]
    _, _, V = np.linalg.svd(A) # EXACT SVD
    p = optimal_probabilities(np.transpose(A), A) # We're going to use to optimal transition probabilities
    kTemp = floor(min(m, n)/4)
    l = 2*kTemp
    cVec = [int(c) for c in set(np.linspace(l, 3*kTemp, num = num_c))]
    cVec.sort()
    # Low dimensional representation via exact SVD
    plot_low(A@np.transpose( V[range(3), :] ), Tags, 'Exact_SVD')
    # Low dimensional representaiton via linear times SVD
    for c in cVec:
        Vlt, _, _ = linearTimesSVD(np.transpose(A), c, 3, p)
        text = 'LinearTimesSVD, c = ' + str(c)
        plot_low( A@Vlt, Tags, text  )
    # Low dimensional representation via Blanczos
    for i_param in i_s:
        text = 'Blanczos, i=' + str(i_param)
        _, _, VBlan = SVD_Blanczos(A, l, i_param)
        plot_low(A@np.transpose(VBlan[range(3), :]), Tags, text)
        
def plotRightSingularVectors(A):
    '''
    For the three different methods (LinearTimeSVD, Blanczos and exact SVD) plot the entries of the top 3 right singular vectors
    '''
    num_c = 2
    i_s = [0, 1, 2]
    m = A.shape[0]
    n = A.shape[1]
    top1RightSV = pd.DataFrame()
    top2RightSV = pd.DataFrame()
    top3RightSV = pd.DataFrame()
    _, _, V = np.linalg.svd(A) # EXACT SVD
    Vt = np.transpose( V[range(3), :] )
    p = optimal_probabilities(np.transpose(A), A) # We're going to use to optimal transition probabilities
    kTemp = floor(min(m, n)/4)
    l = 2*kTemp
    cVec = [int(c) for c in set(np.linspace(l, 3*kTemp, num = num_c))]
    cVec.sort()
    # Low dimensional representation via exact SVD
    top1RightSV['Exact SVD'] = Vt[:, 0]
    top2RightSV['Exact SVD'] = Vt[:, 1]
    top3RightSV['Exact SVD'] = Vt[:, 2]
    # Low dimensional representaiton via linear times SVD
    for c in cVec:
        Vlt, _, _ = linearTimesSVD(np.transpose(A), c, 3, p)
        text = 'LinearTimesSVD, c = ' + str(c)
        top1RightSV[text] = Vlt[:, 0]
        top2RightSV[text] = Vlt[:, 1]
        top3RightSV[text] = Vlt[:, 2]
    # Low dimensional representation via Blanczos
    for i_param in i_s:
        text = 'Blanczos, i=' + str(i_param)
        _, _, VBlan = SVD_Blanczos(A, l, i_param)
        VtBlan = np.transpose(VBlan[range(3), :])
        top1RightSV[text] = VtBlan[:, 0]
        top2RightSV[text] = VtBlan[:, 1]
        top3RightSV[text] = VtBlan[:, 2]
    # We plot them
    # TOP 1
    plt.figure()
    index = range(1, len(top1RightSV)+1)
    cNames = top1RightSV.columns
    plt.plot( index, top1RightSV.iloc[:, 0], c = colormap2(0), label = cNames[0], alpha = 0.8, linewidth = 0.6 )
    plt.plot( index, top1RightSV.iloc[:, 1], c = colormap3(40), label = cNames[1], alpha = 0.7, linewidth = 0.4  )
    plt.plot( index, top1RightSV.iloc[:, 2], c = colormap3(75), label = cNames[2], alpha = 0.7, linewidth = 0.4  )
    plt.plot( index, top1RightSV.iloc[:, 3], c = colormap1(100), label = cNames[3], alpha = 0.5, linewidth = 0.4  )
    plt.plot( index, top1RightSV.iloc[:, 4], c = colormap1(150), label = cNames[4], alpha = 0.5, linewidth = 0.4  )
    plt.plot( index, top1RightSV.iloc[:, 5], c = colormap1(200), label = cNames[5], alpha = 0.5, linewidth = 0.4  )
    plt.ylabel('Right singular vector')
    plt.xlabel('Index')
    plt.title('Top 1 right singular vectors')
    plt.show(block = False)
    plt.legend()
    # TOP 2
    plt.figure()
    index = range(1, len(top2RightSV)+1)
    plt.plot( index, top2RightSV.iloc[:, 0], c = colormap2(0), label = cNames[0], alpha = 0.8, linewidth = 0.6 )
    plt.plot( index, top2RightSV.iloc[:, 1], c = colormap3(50), label = cNames[1], alpha = 0.7, linewidth = 0.4  )
    plt.plot( index, top2RightSV.iloc[:, 2], c = colormap3(75), label = cNames[2], alpha = 0.7, linewidth = 0.4  )
    plt.plot( index, top2RightSV.iloc[:, 3], c = colormap1(100), label = cNames[3], alpha = 0.5, linewidth = 0.4  )
    plt.plot( index, top2RightSV.iloc[:, 4], c = colormap1(150), label = cNames[4], alpha = 0.5, linewidth = 0.4  )
    plt.plot( index, top2RightSV.iloc[:, 5], c = colormap1(200), label = cNames[5], alpha = 0.5, linewidth = 0.4  )
    plt.ylabel('Right singular vector')
    plt.xlabel('Index')
    plt.title('Top 2 right singular vectors')
    plt.show(block = False)
    plt.legend()
    # TOP 3
    plt.figure()
    index = range(1, len(top3RightSV)+1)
    plt.plot( index, top3RightSV.iloc[:, 0], c = colormap2(0), label = cNames[0], alpha = 0.8, linewidth = 0.6 )
    plt.plot( index, top3RightSV.iloc[:, 1], c = colormap3(50), label = cNames[1], alpha = 0.7, linewidth = 0.4 )
    plt.plot( index, top3RightSV.iloc[:, 2], c = colormap3(75), label = cNames[2], alpha = 0.7, linewidth = 0.4 )
    plt.plot( index, top3RightSV.iloc[:, 3], c = colormap1(100), label = cNames[3], alpha = 0.5, linewidth = 0.4 )
    plt.plot( index, top3RightSV.iloc[:, 4], c = colormap1(150), label = cNames[4], alpha = 0.5, linewidth = 0.4 )
    plt.plot( index, top3RightSV.iloc[:, 5], c = colormap1(200), label = cNames[5], alpha = 0.5, linewidth = 0.4 )
    plt.ylabel('Right singular vector')
    plt.xlabel('Index')
    plt.title('Top 3 right singular vectors')
    plt.show(block = False)
    plt.legend()
    # We scatter
    # TOP 1
    plt.figure()
    index = range(1, len(top1RightSV)+1)
    cNames = top1RightSV.columns
    plt.scatter( index, top1RightSV.iloc[:, 0], c = colormap2(0), label = cNames[0], alpha = 0.8, marker = '.', s = [25]*n )
    plt.scatter( index, top1RightSV.iloc[:, 1], c = colormap3(50), label = cNames[1], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top1RightSV.iloc[:, 2], c = colormap3(75), label = cNames[2], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top1RightSV.iloc[:, 3], c = colormap1(100), label = cNames[3], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top1RightSV.iloc[:, 4], c = colormap1(150), label = cNames[4], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top1RightSV.iloc[:, 5], c = colormap1(200), label = cNames[5], alpha = 0.4, marker = '.', s = [5]*n )
    plt.ylabel('Right singular vector')
    plt.xlabel('Index')
    plt.title('Top 1 right singular vectors')
    plt.show(block = False)
    plt.legend()
    # TOP 2
    plt.figure()
    index = range(1, len(top2RightSV)+1)
    plt.scatter( index, top2RightSV.iloc[:, 0], c = colormap2(0), label = cNames[0], alpha = 0.8, marker = '.', s = [25]*n )
    plt.scatter( index, top2RightSV.iloc[:, 1], c = colormap3(50), label = cNames[1], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top2RightSV.iloc[:, 2], c = colormap3(75), label = cNames[2], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top2RightSV.iloc[:, 3], c = colormap1(100), label = cNames[3], alpha = 0.4, marker = '.', s = [5]*n)
    plt.scatter( index, top2RightSV.iloc[:, 4], c = colormap1(150), label = cNames[4], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top2RightSV.iloc[:, 5], c = colormap1(200), label = cNames[5], alpha = 0.4, marker = '.', s = [5]*n )
    plt.ylabel('Right singular vector')
    plt.xlabel('Index')
    plt.title('Top 2 right singular vectors')
    plt.show(block = False)
    plt.legend()
    # TOP 3
    plt.figure()
    index = range(1, len(top3RightSV)+1)
    plt.scatter( index, top3RightSV.iloc[:, 0], c = colormap2(0), label = cNames[0], alpha = 0.8, marker = '.', s = [25]*n )
    plt.scatter( index, top3RightSV.iloc[:, 1], c = colormap3(50), label = cNames[1], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top3RightSV.iloc[:, 2], c = colormap3(75), label = cNames[2], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top3RightSV.iloc[:, 3], c = colormap1(100), label = cNames[3], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top3RightSV.iloc[:, 4], c = colormap1(150), label = cNames[4], alpha = 0.4, marker = '.', s = [5]*n )
    plt.scatter( index, top3RightSV.iloc[:, 5], c = colormap1(200), label = cNames[5], alpha = 0.4, marker = '.', s = [5]*n )
    plt.ylabel('Right singular vector')
    plt.xlabel('Index')
    plt.title('Top 3 right singular vectors')
    plt.show(block = False)
    plt.legend()
 

# A TOY EXAMPLE

# A1 = np.random.randint(-10, 30, size = (100, 500) )
# B1 = np.random.randint(-10, 30, size = (500, 200) )
# exact_mult1 = np.matmul(A1, B1)
# p1 = optimal_probabilities(A1, B1)

# cs = []
# for i in range(100):
#     cs = cs + [(i+1)*2 + 300]*50
# errors_1F = np.zeros((len(cs), 1))
# errors_1S = np.zeros((len(cs), 1))
# for i in range(len(cs)):
#     c = cs[i]
#     C, R = basicMatrixMultiplications(A1, B1, c, p1)
#     approx_multiply = np.matmul(C, R)
#     errors_1F[i] = norm( approx_multiply - exact_mult1, ord = 'fro'  )**2
#     errors_1S[i] = norm( approx_multiply - exact_mult1, ord = 2)**2



# plot_withTheoreticalBounds(A1, B1, cs, errors_1F, errors_1S, p1, 1, logPlot=False, Ylimits=[-5, 5])



# plt.show()


# Example of randomized SVD
# m = 2**6
# Ua = hadamard(m)
# Va = hadamard(2*m)
# s = np.zeros([m,1])
# for k in range(m-1):
#     s[k] = 2/(k+1)
# S = np.zeros([m, 2*m])
# np.fill_diagonal(S, s)
# A = Ua@S@np.transpose(Va)

# plotRightSingularVectors(A)

# time_svd, errors_svd, errors_table_lT, errors_table_B = errorsTable(A)
# print(time_svd)
# print("\n")
# print(errors_svd)
# print("\n")
# print(errors_table_lT)
# print("\n")
# print(errors_table_B)

# Tags = ['one']*m

# LowDimVisualization(A, Tags)

# # plotMethodsErrorsvsExact(A, m, 2*m, 0)

# plt.show()


# l = 2*m
# i_nd = 0
# k = m
# U_B, Sigma_B, V_B = SVD_Blanczos(A, l, i_nd, Sigma_full = False)
# print(U_B.shape)
# print(np.diag(Sigma_B).shape)
# print(V_B.shape)
# er = []
# for i in range(1, m):
#     er.append( np.linalg.norm(  U_B[:, range(i)]@np.diag(Sigma_B[range(i)])@np.transpose(V_B)[range(i), :] - A, ord = 2 )    )
# plt.figure
# plt.plot(range(1,m), er)
# plt.show()
# c = 500
# k = 10
# c = m
# k = 3
# p = optimal_probabilities(A, np.transpose(A))
# ls = []
# rs_1 = []
# rs_2 = []
# UA, SA, VtH = np.linalg.svd(A)
# for j in range(m):
#     H, SigmaC, C = linearTimesSVD(A, c, j, p)
#     ls.append( np.linalg.norm( A - H@np.transpose(H)@A    , ord = 2)**2 )
#     rs_1.append( SA[j]**2 + 2*norm( A@np.transpose(A) - C@np.transpose(C)  , ord = 2)**2 )
#     rs_2.append( SA[j]**2 + norm(A, ord='fro')**2 )
    
# plt.figure()
# plt.plot(range(m), ls)
# plt.show(block=False)
# plt.figure()
# plt.plot(range(m), rs_1)
# plt.show()