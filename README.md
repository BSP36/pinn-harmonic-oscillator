# pinn-harmonic-oscillator
Physics-Informed Neural Network (PINN) to solve Schr&ouml;dinger equations with harmonic oscillator potentials.

**1-dimensional Schr&ouml;dinger equations**
$$\left(-\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)\right)\psi(x) = E\psi(x),$$

with

$$V(x) = \frac{1}{2}\omega^2 x^2.$$

You can choose any other potential by changing the function "potential".




**2-dimensional Schr&ouml;dinger equations**
$$\left(-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial y^2}-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial y^2}+ V(x, y)\right)\psi(x, y) = E\psi(x, y),$$

with

$$V(x, y) = \frac{1}{2}\omega^2 (x^2+y^2).$$

This is implemented in the file "2d.py".
You can easily extend this algorithm to 3-dimensional cases.
