# **V**olume **IN**tegral **E**quation **S**olver (**VINES**) 

VINES is a suite of Python codes for solving volume integral equation formulations of acoustic and electromagnetic scattering problems.

The techniques employed are based on uniform (voxel) spatial discretizations, thus enabling acceleration via the fast-Fourier transform. Furthermore, we include recent preconditioning techniques that yield large reductions in iterative solve times.

The particular scattering scenarios considered when developing this suite are the following:

* Light scattering by atmospheric ice particles
* Light propagation within silicon photonics components
* Acoustic scattering by simple shapes (2D) and (3D)
* High-intensity focused ultrasound in the body

## Bibliography
This work is partially discussed in the following publications:
* *Adiabatic absorbers in photonics simulations with the volume integral equation method*<br>
Alexandra Tambova, Samuel P Groth, Jacob K White, Athanasios G Polimeridis<br>
Journal of Lightwave Technology, 2018
* *Circulant preconditioning in the volume integral equation method for silicon photonics*<br>
Samuel P Groth, Athanasios G Polimeridis, Alexandra Tambova, Jacob K White<br>
Journal of the Optical Society of America A, 2019
* *Accelerating the discrete dipole approximation via circulant preconditioning*<br>
Samuel P Groth, Athanasios G Polimeridis, Jacob K White<br>
Journal of Quantitative Spectroscopy and Radiative Transfer, 2020


