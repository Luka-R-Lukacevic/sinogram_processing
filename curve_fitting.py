"""
Created on Wed Mar  1 09:54:12 2023

@author: lluka
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from helper_functions import find_middle_values
from helper_functions import split_array


def fit_curve(y_data, func= "sin"):
    n = len(y_data)
    x_data = range(n)
    if func == "sin":
        # Define the sin curve function
        def sin_curve(X, a, b, c, d):
            x=X
            return a * np.sin(b * x + c) + d

        # Fit the curve to the data
        params = curve_fit(sin_curve, x_data, y_data, [30,1/100,200,100]) 


        # Get the fitted curve
        fitted_sin = sin_curve(x_data, params[0][0], params[0][1], params[0][2], params[0][3])


        # plot it
        #plt.plot(fitted_sin)
        #plt.plot(y_data)

        #plt.ylim(0,200)
        # Show the plot
        #plt.show()
        return fitted_sin
    
    if func == "poly":
        # Get the fitted curve
        params = np.polyfit(x_data, y_data, deg=10)
        fitted_curve = np.polyval(params, x_data)


        derivative_params = np.polyder(params)
        
        cusps = sorted(np.real(np.roots(derivative_params)))
        
        middle_values = find_middle_values(cusps)
        middle_values = [x for x in middle_values if x > 20]
        middle_values = [x for x in middle_values if x < n - 20]
        multiple_curves_y = split_array(y_data, middle_values)
        multiple_curves_x = split_array(x_data, middle_values)
        
        new_curve = []
        
        for i in range(0,len(middle_values)):
            # Get the fitted curve for the current subarray
            params = np.polyfit(multiple_curves_x[i], multiple_curves_y[i], deg=2)
            new_curve = np.append(new_curve,np.polyval(params, multiple_curves_x[i]))

        #for i in range(80,100):
            #print(new_curve[i],i)
        
        for m in middle_values:
            m = int(round(m)) 
            # Check if m+1 is within the bounds of the array
            if m+1 < len(new_curve):
                d = new_curve[m-1] - new_curve[m]
                h = new_curve[m-1] - new_curve[m-2]
                d+=h
                #print("hiphuraaay",d, m)
            for i in range(m,len(new_curve)):
                # Check if i is within the bounds of the array
                if i < len(new_curve):
                    new_curve[i] += d

            
            
        #plot it
        plt.plot(fitted_curve)
        plt.plot(y_data)
        plt.plot(new_curve)

        plt.ylim(0,300)
        # Show the plot
        plt.show()
        #for i in range(80,100):
            #print(new_curve[i], i)
        return new_curve        
