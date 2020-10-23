#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 bhills <benjaminhhills@gmail.com>
# Distributed under terms of the GNU GPL3.0 license.

"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
July 29, 2019
"""

class constantsHotPointDrill(object):
    """
    Temperature Constants

    Used for Ice Diver modeling
    """
    def __init__(self):
        # general
        self.spy  = 60.*60.*24.*365.24          # Seconds per year (s yr-1)
        self.g = 9.81                           # Gravity (m s-2)
        self.Tf0 = 273.15                       # Reference Tempearature, triple point for water (K)
        self.kBoltz = 1.38064852e-23            # Boltzmann's Constant (m2 kg s-2 K-1)
        self.tol = 1e-5                         # tolerance for numerics
        # ice
        self.rhoi = 917.                        # Ice Density (kg m-3); Cuffey and Paterson (2010) pg. 12
        self.ki = 2.1                           # Thermal Conductivity (J m-1 K-1 s-1); Cuffey and Paterson (2010) pg. 12
        self.ci = 2097.                         # Specific Heat Capacity (J kg-1 K-1); Cuffey and Paterson (2010) pg. 12
        self.L = 3.335e5                        # Latent Heat of Fusion (J kg-1); Cuffey and Paterson (2010) pg. 12
        self.alphai = self.ki/(self.rhoi*self.ci)
        # ethanol
        self.rhoe = 800                         # Density of ethanol (kg m-3) at 0degC; Engineering Toolbox
        self.ke = 0.167                         # Thermal conductivity of ethanol (J m-1 K-1 s-1) at 25degC; Engineerign Toolbox
        self.ce = 2270.                         # Heat capacity of ethanol (J kg-1 K-1) at 0degC; Engineering Toolbox
        self.eta_e = 1.786e-3                   # Dynamic viscosity of ethanol (Pa s) at 0degC; Engineering Toolbox
        self.mmass_e = 46.07                    # Molar mass of ethanol (g/mol)
        self.mol_diff_e = .84e-9                # Molecular diffusivity of aqueous ethanol at 25degC (m2 s-1); Cussler (1997)
        self.rad_e = .22-9                      # Radius of ethanol molecule (m)
        # methanol
        self.rhom = 810                         # Density of ethanol (kg m-3) at 0degC; Engineering Toolbox
        self.km = 0.203                         # Thermal conductivity of ethanol (J m-1 K-1 s-1) at 25degC; Engineering Toolbox
        self.cm = 2400.                         # Heat capacity of ethanol (J kg-1 K-1) at 0degC; Engineering Toolbox
        self.eta_m = .796e-3                    # Dynamic viscosity of ethanol (Pa s) at 0degC (Pa s); Engineering Toolbox
        self.mmass_m = 32.04                    # Molar mass of ethanol (g/mol)
        self.mol_diff_m = .84e-9                # Molecular diffusivity of aqueous ethanol (m2 s-1) at 25degC; Cussler (1997)
        self.rad_m = .21-9                      # Radius of methanol molecule (m)
        # water
        self.rhow = 1000.                       # Density of water (kg m-3)
        self.kw = 0.555                         # Thermal conductivity of water (J m-1 K-1 s-1)
        self.cw = 4212.                         # Heat capacity of water (J kg-1 K-1)
        self.etaw = 1.5e-3                      # Dynamic viscosity of water (Pa s); Engineering Toolbox
        self.mmass_w = 18.02                    # Molar mass of water (g mol-1)
        self.alphaw = self.kw/(self.rhow*self.cw)
        # probe
        self.rho_probe = 8050.                  # Density of probe (kg m-3)
        self.k_probe = 50.2                     # Thermal conductivity of probe (J m-1 K-1 s-1)
        self.c_probe = 466.                     # Heat capacity of probe (J kg-1 K-1)
        self.alpha_probe = self.k_probe/(self.rho_probe*self.c_probe)

        # freezing point depression
        self.Kf = -1.99                         # Freezing point depression constant for ethanol in water
