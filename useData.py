#Several functions to demonstrate some of the results from data.py

import data as da

theData = da.lifetimes('lifetime.txt')
oneD = da.oneD(theData)
twoD = da.twoD(theData)

def showLifetimes():
    '''
    A selection of results from the lifetimes class, 
    Prints:
        
    - The integral of the fit function over a large interval
    
    Plots:
        
    - A selection of fit functions on the raw data histogram
    
    '''
    
    print 'Fit Function integrates to:', theData.integrateFitFunc()
    theData.plotHist()

def showOneD():
    '''
    A selection of results from the oneD class,
    Prints:
        
    - The one-dimensional minimisation parameter value and uncertainties
    - Number of measurements required to obtain S.D. of 10e-15s
    - Fitting parameters for the variation in standard deviation against number of samples
    
    Plots:
        
    - The parabolic approximation for the NLL
    - The variation in standard deviation against number of samples
    
    '''
    oneD.minimise()
    print ''
    oneD.plotMinimise()
    print ''
    oneD.plotAccVary()
    print ''

def showTwoD():
    '''
    A selection of results from the twoD class,
    Prints:
        
    - The two-dimensional minimisation parameter values and uncertainties for each method
    
    Plots:
        
    - Contour plot showing minimisation trajectories 
    - 3D plot showing minimisation trajectories
    
    '''
    twoD.minimise()
    twoD.plotMinContour()
    twoD.plotMinContour(contour=False)
    
    
    
    
    