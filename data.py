import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class lifetimes:
    '''
    Takes the csv data file name as an argument, contains the master data array (self.data) as well as some useful core functions:
    
    - fitFunc(tau, theData, time, sigma, a)
    - plotHist(tau, a)
    - integrateFitFunc(tau)
    - plotFitFunc()
    - logLik(tau, a, theData)
    
    '''
    
    def __init__(self, fileName):
        self.data = np.loadtxt(fileName)

    def fitFunc(self, tau, theData=None, time=None, sigma=None, a=1):
        '''
        Produces fit function on all points from data        
        
        '''
        #Default parameters
        
        if theData is None:
            theData = self.data            
        if time is None:
            time = theData[:,0]
        if sigma is None:
            sigma = theData[:,1]
        
        f = a*(1./(2.*tau))*np.exp(sigma**2./(2.*tau**2.)-time/tau)*sp.erfc((1./2.**0.5)*(sigma/tau - time/sigma)) + (1-a)*(1./(sigma*np.sqrt(2.*np.pi)))*np.exp((-1./2.)*time**2/sigma**2)
        
        return f
        
    def plotHist(self, tau=0.410, a=0.983679420478):
        '''
        Plots histogram of all the data points with the fit function (with optimised parameters) superimposed     
        
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        
        sigma = np.mean(self.data[:,1]) #arithmetic mean of sigma values
        times = np.arange(-2,6,0.02)
        
        ffPerfect = self.fitFunc(tau=tau, sigma=sigma, a=a, time=times)
        ffLow = self.fitFunc(tau=0.2, sigma=sigma, a=a, time=times)
        ffHigh = self.fitFunc(tau=0.6, sigma=sigma, a=a, time=times)
        
        ax1.hist(self.data[:,0], 10e1, normed=1, label='Raw Data', facecolor='green')
        ax1.plot(times,ffLow, color='r', label='Fit Function')
        ax1.set_xlabel('Time / ps')
        ax1.set_ylabel('Probability Density / ps$^{-1}$')
        ax1.set_title('(a) Tau = 0.2 ps')
        ax1.legend()
        
        ax2.hist(self.data[:,0], 10e1, normed=1, label='Raw Data', facecolor='green')
        ax2.plot(times,ffPerfect, color='r', label='Fit Function')
        ax2.set_xlabel('Time / ps')
        ax2.set_ylabel('Probability Density / ps$^{-1}$')
        ax2.set_title('(b) Tau = 0.410 ps')
        
        ax3.hist(self.data[:,0], 10e1, normed=1, label='Raw Data', facecolor='green')
        ax3.plot(times,ffHigh, color='r', label='Fit Function')
        ax3.set_xlabel('Time / ps')
        ax3.set_ylabel('Probability Density / ps$^{-1}$')
        ax3.set_title('(c) Tau = 0.6 ps')
        
        
    
    def integrateFitFunc(self, tau=0.4):
        '''
        Integrates the Fit Function (with arbitrary tau) to check that it is normalised
        
        '''
        
        times = np.arange(-50.,50.0,0.02)
        pdfVals = self.fitFunc(tau=0.4,sigma=0.1,time=times)
        
        integral = np.trapz(pdfVals, times)
            
        return integral
        
    
    def plotFitFunc(self):
        '''
        Plots points for the fit function at varying tau
        
        '''
        times = np.arange(-1.0,3,0.05)
        sigmas = self.data[:,1]
        coords = []
        
        for time in times:
            for sigma in sigmas:
                prob = self.fitFunc(tau=0.40454185055, theData = self.data, time=time, sigma=sigma)
                coords.append([time,prob])
                
        coords = np.array(coords)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(coords[:,0],coords[:,1], s=0.02)
        ax.set_ylabel('PDF')
        ax.set_xlabel('Tau')
        
    def logLik(self, tau, a=1, theData=None):
        '''
        Produces negative Log-likelihood from data and a given parameter value tau
        '''
        if theData is None:
            theData = self.data  
    
        NLL = -np.sum(np.log(self.fitFunc(tau, theData=theData,a=a)))
    
        return NLL
        

class oneD:
    '''
    Takes a lifetimes object argument, contains functions for the one-dimensional minimisation of the NLL to find the most likely tau parameter value:
        
    - secAcc(tauCen, tauMin, tauMax, thresh, theData)
    - bisecAcc(tauCen, tauMin, tauMax, thresh, theData)
    - curvAcc(tau0,tau1,tau2,y0,y1,y2, return_coeffs)
    - minimise(tau0, tau1, tau2, accuracy, theData, newtAccs, newtAccsOnly, tau_finals)
    - plotMinimise(theData)
    
    To analyse the change in standard deviation with sample size
    
    - sampleNewtAccVary()
    - sampleCurvAccVary()
    - sampleVaryFitFunc(n, a, b)
    - invSampleVaryFitFunc(sd, a, b)
    - findRegress(coords)
    - plotAccVary()
    
    '''
    
    def __init__(self, dataOb):
        self.data = dataOb.data
        self.dataOb = dataOb

    def secAcc(self, tauCen, tauMin, tauMax, thresh=1e-10, theData=None):
            '''
            Returns the standard deviation when given a certain range using the Secant method         
            
            '''
            if theData is None:
                theData = self.data
            
            def f(tau):
                return self.dataOb.logLik(tau, theData=theData)-self.dataOb.logLik(tauCen, theData=theData)-0.5
            
            err = 1.
            t0,t1 = tauMin, tauMax
            
            while err > thresh: 
                
                t2 = t1 - f(t1)*(t1-t0)/(f(t1)-f(t0))
                t0 = t1
                t1 = t2
                
                err = abs((t1-t0))/2.
            
            acc = abs(tauCen-t2)
            
            return acc
            
    def bisecAcc(self, tauCen, tauMin, tauMax, thresh=1e-10, theData=None):
            '''
            Returns the standard deviation when given a certain range using the bisection method         
            
            '''
            if theData is None:
                theData = self.data
            
            def f(tau):
                return self.dataOb.logLik(tau, theData=theData)-self.dataOb.logLik(tauCen, theData=theData)-0.5
            
            err = 1.
            t0,t1 = tauMin, tauMax
            t2 = (t0+t1)/2.
            
            while err > thresh:
                
                if f(t2) == 0:
                    return t2
                elif f(t0)*f(t2) < 0:
                    t1 = t2
                else:
                    t0 = t2
                t2 = (t0+t1)/2.
                err = abs((t1-t0)/2)
            
            acc = abs(tauCen-t2)
            
            return acc
    
    
    def curvAcc(self, tau0,tau1,tau2,y0,y1,y2, return_coeffs=False):
            '''
            Returns value of standard deviation from the second derivative of the parabolic approx. for the NLL            
            
            '''
            L = np.array([[tau0**2, tau0, 1],[tau1**2, tau1, 1],[tau2**2, tau2, 1]]) 
            LInv = np.linalg.inv(L)
            Y = np.array([y0,y1,y2])
            D = np.dot(LInv,Y)

            acc = np.sqrt(1/(2*D[0]))            
            
            if return_coeffs is False:
                return acc
            
            else:
                return D

    def minimise(self, tau0=0.38, tau1=0.4, tau2=0.42, accuracy=1e-10, theData=None, newtAccs = True, newtAccsOnly=False, tau_finals=False):
        '''
        Takes three initial tau values and finds the minimum of the NLL for the data using a parabolic minimiser and the standard deviation for this value using both the curvature analysis and Secant methods
        
        '''
        if theData is None:
            theData = self.data
        
        #Minimisation
    
        dtau3 = 0.5
        
        y0 = self.dataOb.logLik(tau0, theData=theData)
        y1 = self.dataOb.logLik(tau1, theData=theData)
        y2 = self.dataOb.logLik(tau2, theData=theData)
        oldtau3 = 0
        
        while abs(dtau3) > accuracy:
        
            tau3 = 0.5*((tau2**2-tau1**2)*y0+(tau0**2-tau2**2)*y1+(tau1**2-tau0**2)*y2)/((tau2-tau1)*y0+(tau0-tau2)*y1+(tau1-tau0)*y2)
            
            dtau3 = abs(tau3-oldtau3)   
            
            y3 = self.dataOb.logLik(tau3, theData=theData)
            
            ys = np.array(([tau0,y0],[tau1,y1],[tau2,y2],[tau3,y3]))
            ys = ys[ys[:,1].argsort()]
            ys = ys[:3]
            ys = ys[ys[:,0].argsort()]
            
            tau0 = ys[:,0][0]
            tau1 = ys[:,0][1]
            tau2 = ys[:,0][2]
            y0 = ys[:,1][0]
            y1 = ys[:,1][1]
            y2 = ys[:,1][2]
            
            oldtau3 = tau3
        
        #Outputs
        
        if tau_finals:
            return [tau0,tau3,tau2]
        
        if newtAccs:
            seccAccMin = self.secAcc(tau3,0.39,tau3, theData=theData)
            seccAccMax = self.secAcc(tau3,tau3,0.42, theData=theData)
            curvAcc = self.curvAcc(tau0,tau1,tau2,y0,y1,y2)
            self.min1D = tau3, curvAcc, seccAccMin, seccAccMax 
            print 'Tau Min: ', tau3
            print ''
            print 'Standard Deviations'
            print 'Curvature: ', curvAcc
            print 'Secant (tau < Tau Min): ', seccAccMin
            print 'Secant (tau > Tau Min): ', seccAccMax
            print ''
        
        if newtAccsOnly:
            newtAccMin = self.secAcc(tau3,0.3,tau3, theData=theData)
            newtAccMax = self.secAcc(tau3,tau3,0.5, theData=theData)
            newtAcc = (newtAccMin+newtAccMax)/2. #average newton accuracies
            return newtAcc
        
        if not newtAccs:
            curvAcc = self.curvAcc(tau0,tau1,tau2,y0,y1,y2)
            return tau3, curvAcc
    
    def plotMinimise(self, theData = None):
        '''
        Plots the NLL and the parabolic approximations
        
        '''
        if theData is None:
            theData = self.data
        
        taus = np.arange(0.398,0.410, 0.00001)
        
        #Accurate Solution for NLL
        
        NLLs = []
        
        for tau in taus:
            NLLs.append(self.dataOb.logLik(tau))
            
        NLLs = np.array(NLLs)
        
        #Parabolic Approximation

        tau0, tau1, tau2 = self.minimise(tau_finals=True)
        y0, y1, y2 = self.dataOb.logLik(tau0, theData=theData), self.dataOb.logLik(tau1, theData=theData), self.dataOb.logLik(tau2, theData=theData)
        coeffs2 = self.curvAcc(tau0,tau1,tau2,y0,y1,y2,return_coeffs=True)
        def para2(tau):
            return coeffs2[0]*tau**2 + coeffs2[1]*tau + coeffs2[2]
        para2 = para2(taus)
        
        tauMin, curvAcc = self.minimise(newtAccs=False)
        NLLMin = self.dataOb.logLik(tauMin)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)  
        ax.plot(taus, NLLs, label='NLL solution')
        #ax.plot(taus, para1, linestyle='dashed', label='Initial Parabolic Approximation')
        ax.plot(taus, para2, linestyle='dashed', label='Final Parabolic Approximation')
        ax.axhline(y=NLLMin+0.5, color='k', linestyle = 'dashdot', linewidth=2, label='Minimum NLL + 0.5')
        ax.axvline(x=tauMin+curvAcc, color='r', linestyle = 'dashdot', linewidth=1)
        ax.axvline(x=tauMin-curvAcc, color='r', linestyle = 'dashdot', linewidth=1)
        #ax.axvline(x=tauMin, linestyle = 'dotted', linewidth=1, color='k') #Minimum line
        ax.set_xlabel('Tau / ps')
        ax.set_ylabel('NLL')
        ax.legend()
    
    def sampleNewtAccVary(self):
        '''
        Returns coordinates for the variation of accuracy in tau with sample size in 1D using newtAcc
        
        '''
        
        accCoords = []
        ns = np.arange(10,10010,100)
        
        for n in ns:
            acc = self.minimise(theData=self.data[:n], newtAccs = False, newtAccsOnly=True)
            accCoords.append([n, acc])
        
        return np.array(accCoords)

    def sampleCurvAccVary(self):
        '''
        Returns coordinates for the variation of accuracy in tau with sample size in 1D using curvAcc
        
        '''

        accCoords = []
        ns = np.arange(10,10010,100)
        
        for n in ns:
            acc = self.minimise(theData=self.data[:n], newtAccs = False)[1]
            accCoords.append([n, acc])
        
        return np.array(accCoords)
    
    def sampleVaryFitFunc(self, n, a, b):
        '''
        Expected form of the variation in standard deviation with sample size n
        
        '''
        
        return float(a)*n**(-b)
        
    def invSampleVaryFitFunc(self, sd, a, b):
        '''
        The inverse of our expected form of the variation in standard deviation with sample size n.
        This will return the value of n at which a certain error would be found
        
        '''
        
        return (a/sd)**(1./b)
    
    def findRegress(self, coords):
        '''
        Fits a curve of the form sampleVaryFitFunc() to given coords and returns the fit parameters
        
        '''
        
        samples = coords[:,0]
        sds = coords[:,1]
        mask = ~np.isnan(samples) & ~np.isnan(sds)#mask nan values
        n = samples[mask]
        sds = sds[mask]
        self.regress = curve_fit(self.sampleVaryFitFunc, n, sds)
        
        return self.regress
        
        
    def plotAccVary(self):
        '''
        Plots the varitaion of accuracy in tau (in picoseconds) with sample size in 1D for the Newton-Raphson and Curvature methods as well as a regression fit for the Newton Raphson Method
        Returns number of samples (to the nearest one) required to reach 10e-15 s accuracy and the error from the fit statistics
        
        '''
        newtAccVary = self.sampleNewtAccVary()
        curvAccVary = self.sampleCurvAccVary()
        
        coordsList = np.array([newtAccVary, curvAccVary])
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111) 
        
        #plot S.D. variations
        
        ax1.plot(coordsList[0][:,0],coordsList[0][:,1], label = 'Secant S.D. Variation')
        ax1.plot(coordsList[1][:,0],coordsList[1][:,1], label = 'Curvature S.D. Variation')
            
        #Find regression line from Secant points
                
        regress = self.findRegress(coordsList[0])
        a = regress[0][0]
        b = regress[0][1]
        sampleThresh = int(self.invSampleVaryFitFunc(1e-3,a, b))
        ns = np.arange(10,sampleThresh+1e3,100)
        sds = self.sampleVaryFitFunc(ns,a,b)
        perr = np.sqrt(np.diag(regress[1]))
        self.perr = perr
        upper_bound = self.invSampleVaryFitFunc(1e-3, a+perr[0], b-perr[1])
        lower_bound = self.invSampleVaryFitFunc(1e-3, a-perr[0], b+perr[1])
        error = int((upper_bound-lower_bound)/2.) #average error

        #plot regression line
        
        ax1.plot(ns,sds,label='Secant Fit')
        ax1.axhline(y=1e-3, color='k', linestyle = 'dashed', linewidth=2)
        ax1.axvline(x=sampleThresh, color='r', linestyle = 'dashed', linewidth=1)
            
        ax1.set_yscale("log", nonposy='clip')
        ax1.set_ylabel('Standard Deviation / ps')
        ax1.set_xscale("log", nonposy='clip')
        ax1.set_xlabel('Sample size')
        ax1.legend()
        
        print 'Number of measurements to obtain S.D. of 10e-15s: ', sampleThresh, ' +/- ', error
        print 'Fit exponent of n (b): ', regress[0][1], ' +/- ', perr[1]
        print 'Fit constant (k): ', regress[0][0], ' +/- ', perr[0]
            
        
class twoD:
    '''
    Takes a lifetimes object argument, contains functions for the two-dimensional minimisation of the NLL to find the most likely values of tau and a:
    
    - gradNLL(tau, a, h)
    - hessNLL(tau, a, h)
    - gradMin(tauInit, aInit, h, alpha, produceVal)
    - quasiNewtMin(tauInit, aInit, h, alpha, produceVal)
    - newtMin(tauInit, aInit, h, alpha, produceVal)
    - secAcc(tauCen, tauMin, tauMax, aCen, aMin, aMax, thresh, theData)
    - meanSecAcc(tauCen, tauMin, tauMax, aCen, aMin, aMax, thresh, theData)
    - covMatrix(tau, a, h)
    - covError(tau, a, h)
    - corrCoeff(tau, a, h)
    - minimise()
    - plotMinContour(contour)
    
    '''
    
    def __init__(self, dataOb):
        self.data = dataOb.data
        self.dataOb = dataOb
    
    def gradNLL(self,tau,a,h):
        '''
        Finds gradient of NLL function in 2D
        
        '''
        b = self.dataOb.logLik(tau,a)
        dNLLdtau = (self.dataOb.logLik(tau+h,a)-b)/h
        dNLLda = (self.dataOb.logLik(tau,a+h)-b)/h
        dNLL = np.array([dNLLdtau, dNLLda])
        return dNLL
    
    def hessNLL(self,tau,a,h):
        '''
        Returns Hessian matrix for the NLL at a given tau and a
        
        '''
        b = self.gradNLL(tau,a,h)
        d2tau2 = (self.gradNLL(tau+h,a,h)[0]-b[0])/h
        d2a2 = (self.gradNLL(tau,a+h,h)[1]-b[1])/h
        d2atau = (self.gradNLL(tau,a+h,h)[0]-b[0])/h
        H = np.array([[d2tau2,d2atau],[d2atau,d2a2]])
        return H
        
    def gradMin(self, tauInit=0.4, aInit=0.9, h=0.00002, alpha=0.00003, produceVal = True):
        '''
        Finds minimum of NLL using the Gradient method, also stores the coordinates of the route taken as self.gradMinLine
        
        '''
        h = float(h)
        coords = [[tauInit, aInit, self.dataOb.logLik(tauInit, aInit)]]
        x0 = np.array([tauInit, aInit])
        errTau, erra = 1, 1
        
        while errTau > 1e-10 or erra > 1e-10:
            dNLL = self.gradNLL(x0[0], x0[1], h)
            x1 = x0 - alpha*dNLL
            errTau = abs(x0[0] - x1[0])
            erra = abs(x0[1] - x1[1])
            x0 = x1
            coords.append([x0[0],x0[1], self.dataOb.logLik(x0[0],x0[1])])
        
        self.gradMinLine = np.array(coords)
        
        self.min2DGrad = x1
        
        if produceVal is True:
            return x1
    
    def quasiNewtMin(self, tauInit=0.4, aInit=0.9, h=0.00002, alpha=0.00003, produceVal = True):
        '''
        Finds minimum of NLL using the Quasi-Newton method, also stores the coordinates of the route taken as self.quasiMinLine
        
        '''
        
        h = float(h)
        coords = [[tauInit, aInit, self.dataOb.logLik(tauInit, aInit)]]
        errTau, erra = 1, 1
        G0 = np.identity(2)
        x0 = np.array([tauInit, aInit])
        errTau, erra = 1, 1
        
        while errTau > 1e-10 or erra > 1e-10:
            df = self.gradNLL(x0[0],x0[1],h)
            x1 = x0 - alpha*np.dot(G0,df)
            delta = x1 - x0
            gamma = self.gradNLL(x1[0],x1[1],h)-df
            G1 = G0 + np.outer(delta,delta)/np.dot(gamma,delta) - (np.dot(G0,np.dot(G0,np.outer(delta,delta))))/np.dot(gamma,np.dot(G0,gamma))
            G0 = G1
            errTau = abs(x0[0] - x1[0])
            erra = abs(x0[1] - x1[1])
            x0 = x1
            coords.append([x1[0],x1[1], self.dataOb.logLik(x1[0], x1[1])])
        
        self.quasiMinLine = np.array(coords)
        
        self.min2DQuasi = x0
        
        if produceVal == True:
            return x0
    
    def newtMin(self, tauInit=0.4, aInit=0.9, h=0.00002, alpha=0.00003, produceVal=True):
        '''
        Finds minimum of NLL using the Newton method, also stores the coordinates of the route taken as self.newtMinLine
        
        '''
        
        h = float(h)
        coords = [[tauInit, aInit, self.dataOb.logLik(tauInit, aInit)]]
        errTau, erra = 1, 1
        x0 = np.array([tauInit, aInit])
        errTau, erra = 1, 1
        
        while errTau > 1e-10 or erra > 1e-10:
            df = self.gradNLL(x0[0],x0[1],h)
            H = self.hessNLL(x0[0],x0[1],h)
            Hinv = np.linalg.inv(H)
            x1 = x0 - np.dot(Hinv,df)
            errTau = abs(x0[0] - x1[0])
            erra = abs(x0[1] - x1[1])
            x0 = x1
            coords.append([x1[0],x1[1], self.dataOb.logLik(x1[0], x1[1])])
        
        self.newtMinLine = np.array(coords)
        
        self.min2DNewt = x0
        
        if produceVal == True:
            return x0
    
    def secAcc(self, tauCen=0.409674421874, tauMin=0.4, tauMax=0.409674421874, aCen=0.983679421758, aMin=0.97, aMax=0.983679421758, thresh=1e-10, theData='data'):
            '''
            Returns the standard deviation in 2D (formatted as [Error in Tau, Error in a] either one evaluated with the other parameter held at the minimum) when given a certain range using the Newton-Raphson method         
            Note: this method is not as accurate as the covariance method (featured later) as it ignores any correlation between the two parameters, and hence isn't used in the results.
            
            '''
            if theData == 'data':
                theData = self.data
            
            def f(tau, a): # = zero when tau is at one standard deviation from tauCen
                return self.dataOb.logLik(tau=tau,a=a,theData=theData)-self.dataOb.logLik(tau=tauCen,a=aCen, theData=theData)-0.5
            
            #Tau error
            
            errTau = 1.
            t0,t1 = tauMin, tauMax
            
            while errTau > thresh:
                
                t2 = t1 - f(t1,aCen)*(t1-t0)/(f(t1,aCen)-f(t0,aCen))
                t0 = t1
                t1 = t2
                
                errTau = abs((t1-t0)/2)
                
            #a error
            
            erra = 1.
            a0,a1 = aMin, aMax
            
            while erra > thresh:
                
                a2 = a1 - f(tauCen,a1)*(a1-a0)/(f(tauCen,a1)-f(tauCen,a0))
                a0 = a1
                a1 = a2
                
                erra = abs((a1-a0)/2)
            
            accTau = abs(tauCen-t2)
            acca = abs(aCen-a2)
            
            return np.array([accTau,acca])
    
    def meanSecAcc(self, tauCen=0.40967442187827369, tauMin=0.4, tauMax=0.5, aCen=0.98367942170760869, aMin=0.97, aMax=0.99, thresh=1e-10, theData='data'):
        '''
        Returns mean of standard deviation 'above' and 'below' minimum
        
        '''
        accAv = (self.secAcc(tauCen=tauCen, tauMin=tauMin, tauMax=tauCen, aCen=aCen, aMin=aMin, aMax=aCen)+self.secAcc(tauCen=tauCen, tauMin=tauCen, tauMax=tauMax, aCen=aCen, aMin=aCen, aMax=aMax))/2.
        
        return accAv
    
    def covMatrix(self, tau, a, h=0.000002):
        '''
        Returns the covariance matrix in tau and a
        
        '''
        
        cov = np.linalg.inv(self.hessNLL(tau=tau, a=a, h=h))
        
        return cov
    
    def covError(self, tau, a, h=0.000002):
        '''
        Takes the covariance matrix as an argument and returns the error in tau and a parameters
        
        '''
        covMatrix = self.covMatrix(tau,a,h)
        
        perr = np.sqrt(np.diag(covMatrix))
        
        return perr
    
    def corrCoeff(self, tau, a, h=0.000002):
        '''
        Takes the covariance matrix as an argument and returns the correlation coefficient between tau and a parameters
        
        '''
        covMatrix = self.covMatrix(tau,a,h)
        
        perr = np.sqrt(np.diag(covMatrix))
        
        corrCoeff = covMatrix[0][1]/perr[0]/perr[1]
        
        return corrCoeff
    
    
    def minimise(self):
        '''
        Returns 2D Minimum points and their associated standard deviations for each of the three methods
        
        '''
        
        gradMin = self.gradMin()
        quasiNewtMin = self.quasiNewtMin()
        newtMin = self.newtMin()
        
        gradMin = [gradMin, self.covError(tau=gradMin[0],a=gradMin[1])]
        quasiNewtMin = [quasiNewtMin, self.covError(tau=quasiNewtMin[0],a=quasiNewtMin[1])]
        newtMin = [newtMin, self.covError(tau=newtMin[0],a=newtMin[1])]
        
        print 'Gradient Method'
        print 'Min tau:', gradMin[0][0], ' +/- ', gradMin[1][0]
        print 'Min a:', gradMin[0][1], ' +/- ', gradMin[1][1]
        print ''
        print 'Quasi-Newton Method'
        print 'Min tau:', quasiNewtMin[0][0], ' +/- ', quasiNewtMin[1][0]
        print 'Min a:', quasiNewtMin[0][1], ' +/- ', quasiNewtMin[1][1]
        print ''
        print 'Newton Method'
        print 'Min tau:', newtMin[0][0], ' +/- ', newtMin[1][0]
        print 'Min a:', newtMin[0][1], ' +/- ', newtMin[1][1]
        
        
    def plotMinContour(self, contour=True):
        '''
        Plots the 3D Surface for the NLL close the minimum and the trajectories of each minimising method
        
        '''
        #Evaluate the minimiser methods
        
        self.gradMin(produceVal=False)
        self.quasiNewtMin(produceVal=False)
        minimum = self.newtMin()
        
        #The minimisation trajectories for each method
        
        minLines = [{'label':'Gradient Method','coords':self.gradMinLine,'color':'b'},{'label':'Quasi-Newton Method','coords':self.quasiMinLine,'color':'g'},{'label':'Newton Method','coords':self.newtMinLine,'color':'r'}]
        
        #Produce points for the NLL surface
        
        minLine = minLines[0]['coords']
        bounds = [[min(minLine[:,0]),max(minLine[:,0])],[min(minLine[:,1]),max(minLine[:,1])]]
        
        t = np.arange(bounds[0][0],(bounds[0][1]-bounds[0][0])/5.+bounds[0][1],0.001)
        a = np.arange(bounds[1][0],1,0.001)
        T, A = np.meshgrid(t,a)
        self.T, self.A = T, A
        Z = np.ones(T.shape)
        for m in range(len(T)):
            for n in range(len(T[m])):
                Z[m][n] = self.dataOb.logLik(T[m][n],A[m][n])
        
        #Contour Plot
                
        if contour:
            
            plt.figure()
            cont = plt.contour(T,A,Z,10)
            plt.clabel(cont, inline=1, fontsize=10)
            for line in minLines:
                plt.plot(line['coords'][:,0],line['coords'][:,1], label=line['label'], linestyle='dashed')
            plt.plot(minimum[0],minimum[1], marker='o', markersize=7, fillstyle = 'none', color='k')
            plt.xlabel('Tau / ps')
            plt.ylabel('a')
            plt.legend()
            plt.show()
            
        #3D Plot
        
        else:
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #surf = ax.plot_surface(T, A, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)              
            surf = ax.plot_wireframe(T, A, Z, rstride=5, cstride=5, color='k')
            ax.set_xlabel('Tau / ps')
            ax.set_ylabel('a')
            ax.set_zlabel('NLL')
            for line in minLines:
                ax.plot(line['coords'][:,0],line['coords'][:,1], line['coords'][:,2], label=line['label'],color=line['color'])
            ax.legend()      

#Define objects      
           
#data = lifetimes('lifetime.txt')
#oneD = oneD(data)
#twoD = twoD(data)

#corrCoeff = twoD.corrCoeff(tau=0.40967442193,a=0.983679421611)
