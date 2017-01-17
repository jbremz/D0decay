—- Jim Bremner — 
—- Lifetime of a D0 Meson —

———————————————————

Files:

data.py - Contains classes for analysis of the lifetime.txt file

lifetime.txt - The particle decay time data in 10,000 pairs of (measured decay time, uncertainty) in ps 

useData.py - Contains functions for demonstrating the results from data.py [as seen in the report]

Jim Bremner - A Log Likelihood fit for extracting the D0 lifetime.pdf - the final report detailing the methods. Read for a more general overview of the aims and results.

NOTE: See comments at bottom of data.py to reproduce the correlation coefficient

———————————————————

—> data.py

Class:

> lifetimes(fileName) - Takes the csv data file name as an argument, contains the master data array (self.data) as well as some useful core functions

Functions:

 - fitFunc(tau, theData, time, sigma, a)
 - plotHist(tau, a)
 - integrateFitFunc(tau)
 - plotFitFunc()
 - logLik(tau, a, theData)

Class: 

> oneD(dataOb) - Takes a lifetimes object argument, contains functions for the one-dimensional minimisation of the NLL to find the most likely tau parameter value

Functions:

 - secAcc(tauCen, tauMin, tauMax, thresh, theData)
 - bisecAcc(tauCen, tauMin, tauMax, thresh, theData)
 - curvAcc(tau0,tau1,tau2,y0,y1,y2, return_coeffs)
 - minimise(tau0, tau1, tau2, accuracy, theData, newtAccs, newtAccsOnly, tau_finals)
 - plotMinimise(theData)
    
To analyse the change in standard deviation with sample size:
    
 - sampleNewtAccVary()
 - sampleCurvAccVary()
 - sampleVaryFitFunc(n, a, b)
 - invSampleVaryFitFunc(sd, a, b)
 - findRegress(coords)
 - plotAccVary()

Class:

> twoD(dataOb) - Takes a lifetimes object argument, contains functions for the two-dimensional minimisation of the NLL to find the most likely values of tau and a

Functions:

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

——————————————————————

—> useData.py

Functions:

 - showLifetimes() - A selection of results from the lifetimes class
 - showOneD() - A selection of results from the oneD class
 - showTwoD() - A selection of results from the twoD class
    


