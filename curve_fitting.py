import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



def fit_curve(y_data, func= "poly"):
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
        plt.plot(fitted_sin, label='Corrected Center')
        plt.plot(y_data, label='Noisy Center')
        plt.ylim(0, 100)
        plt.xlabel('Angle in degrees')
        plt.ylabel('Center')
        plt.legend()
        plt.show()


        return fitted_sin
    
    if func == "poly":
        # Get the fitted curve
        params = np.polyfit(x_data, y_data, deg=3)
        fitted_curve = np.polyval(params, x_data)

        plt.plot(y_data, label='Noisy Center')
        plt.plot(fitted_curve, label='Corrected Center')
        plt.ylim(0, 100)
        plt.xlabel('Angle in degrees')
        plt.ylabel('Center')
        plt.legend()
        plt.show()
        
        return fitted_curve        
