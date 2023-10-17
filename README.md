# CT Object Reconstruction and Processing

This project contains a set of Python scripts for creating a CT (Computed Tomography) Object, reconstructing images and applying various processing techniques. The code provided here allows to simulate a CT object, add shapes to the image, generate sinograms, perform image reconstructions, and process the resulting images.

## Contents:

1. Prerequisites
2. Files Description
3. Usage
4. Examples


## Prerequisites:

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- numpy
- matplotlib
- skimage
- scipy

You can install the required dependencies using the following command:

pip install numpy matplotlib scikit-image scipy

## Files Description:

The project consists of the following files:

1. ct_object.py: Defines the CT_Object class, which represents a CT object. It allows you to add ellipses or rectangles to the image, generate sinograms, perform image reconstructions, and apply noise/perturbation to the sinogram.

2. curve_fitting.py: Contains the fit_curve function, which fits a curve to the given data points using either a polynomial or sine function.

3. helper_functions.py: Contains utility functions used in ct_object.py and processing_part_circle.py.

4. tych_radon_math.py: Provides functions for regularized image reconstruction using the Radon transform.

5.-7. Processing_part_circle.py, Reconstruction_part_circle.py, Reconstruction_part_circle_noise.py: Demonstrates image reconstruction and processing using the CT_Object class. It shows how to add circles to the image, generate sinograms, perturb sinograms with noise, and reconstruct the images using curve fitting.
Calculates MSE and EMD values featured in the tables for the circle.

8.-10. Processing_part_rectangle.py, Reconstruction_part_rectangle.py, Reconstruction_part_rectangle_noise.py: See above, only for the rectangle.

11.-13. Processing_part_two_circles.py, Reconstruction_part_two_circles.py, Reconstruction_part_two_circles_noise.py: See above, only for the two circles.

These scripts are part of the CT Object Reconstruction and Processing bachelors thesis and are used to demonstrate various image reconstruction and processing techniques. Each script contains code to perform specific tasks related to the thesis.

## Usage:

To use this project, follow these steps:

1. Install the prerequisites mentioned in the Prerequisites section.

2. Import the CT_Object class and other necessary functions in your Python script.

3. Create an instance of CT_Object with an initial image.

4. Use the methods of the CT_Object class to add shapes, generate sinograms, perform reconstructions, and apply processing techniques.
