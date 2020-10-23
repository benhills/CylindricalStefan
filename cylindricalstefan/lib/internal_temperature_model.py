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
October 23, 2020
"""

import numpy as np

from cylindricalstefan.lib.constants import constantsHotPointDrill
const = constantsHotPointDrill()
from cylindricalstefan.lib.analytical_pure_solution import analyticalMelt

import dolfin

# --------------------------------------------------------------------------------------------

class internal_temperature_model():
    """
    #TODO: add a description
    """

    def __init__(self,const=const):
        """
        Initial Variables
        """

        # Temperature Variables
        self.T_inf = -20.                       # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature
        self.Q_wall = 0.0                       # Heat Source after Melting (W)
        self.Q_initialize = 2500.               # Heat Source for Melting (W)

        # Domain Specifications
        self.R_center = 0.001                    # Inner Domain Edge (m)
        self.R_inf = 1.                         # Outer Domain Edge (m)
        self.R_melt = 0.04                      # Melt-Out Radius (m)
        self.n = 100                            # Mesh resolution
        self.dt = 10.                           # Time step (s)
        self.t_final = 2.*3600.                 # End simulation time (s)

        # Flags to keep operations in order
        self.flags = []

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def log_transform(self,const=const):
        """
        Nondimensionalize and transform to logarithmic coordinates.
        This puts most of the points near the borehole wall and actually makes the
        math more like the cartesian diffusion problem.
        """

        # Nondimensionalize (Humphrey and Echelmeyer, 1990)
        self.Tstar = self.T_inf/abs(self.T_inf)                                 # Dimensionless temperature
        self.Rstar = self.R_melt/self.R_melt                                    # Dimensionless melt-out radius
        self.Rstar_center = self.R_center/self.R_melt                                 # Dimensionless outer domain edge
        self.Rstar_inf = self.R_inf/self.R_melt                                 # Dimensionless outer domain edge
        self.Qstar = self.Q_wall/(2.*np.pi*const.ki*abs(self.T_inf))            # Dimensionless heat source for melting
        Lv = const.L*const.rhoi                                                 # Latent heat of fusion per unit volume
        self.astar_i = Lv/(const.rhoi*const.ci*abs(self.T_inf))                 # Thermal diffusivity of ice
        diff_ratio = (const.k_probe*const.rhoi*const.ci)/(const.ki*const.rho_probe*const.c_probe)
        self.astar_probe = diff_ratio*self.astar_i                              # Thermal diffusivity of probe
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_melt**2.     # Characteristic time (~freeze time)

        # Tranform to a logarithmic coordinate system so that there are more points near the borehole wall.
        self.w0 = np.log(self.Rstar_center)                                            # Log dimensionless inner domain edge
        self.w_melt = np.log(self.Rstar)                                            # Log dimensionless melt-out radius
        self.wf = np.log(self.Rstar_inf)                                        # Log dimensionless outer domain edge

        self.flags.append('log_transform')

    def get_domain(self):
        """
        Define the Finite Element domain for the problem
        """

        # Finite Element Mesh in solid
        self.mesh = dolfin.IntervalMesh(self.n,self.w0,self.wf)
        self.V = dolfin.FunctionSpace(self.mesh,'CG',1)
        self.coords = self.V.tabulate_dof_coordinates().copy()
        # TODO: is this used?
        self.idx_probe = np.argmin(abs(self.Rstar-self.coords))

        # TODO: define a variable diffusivity for the probe/ice
        self.astar = dolfin.project(dolfin.Expression('astar_i',degree=1,astar_i=self.astar_i),self.V)

        self.flags.append('get_domain')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def get_initial_conditions(self,rho_solute=const.rhom,data_dir=None):
        """
        Set the initial condition at the end of melting (melting can be solved analytically)
        """

        # --- Initial states --- #
        # ice temperature
        self.u0 = dolfin.Function(self.V)
        T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.coords[:,0])*self.R_melt,self.T_inf,self.Q_initialize,R_target=self.R_melt)
        self.u0.vector()[:] = T/abs(self.T_inf)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(self.t_melt,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0

        # --- Define the test and trial functions --- #
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)
        self.T = dolfin.Function(self.V)

        self.flags.append('get_ic')

    def get_boundary_conditions(mod):
        """
        Define Boundary Conditions
        """

        # Right boundary is the far-field temperature
        class Inf(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                    return on_boundary and x[0] > mod.wf - const.tol

        # Initialize boundary classes
        mod.Inf = Inf()
        # Set the Dirichlet Boundary condition at
        mod.bc_inf = dolfin.DirichletBC(mod.V, mod.Tstar, mod.Inf)

        mod.flags.append('get_bc')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def solve_thermal(self):
        """
        Solve the thermal diffusion problem.
        """

        # Set up the variational form for the current mesh location
        F = (self.u-self.u0)*self.v*dolfin.dx + self.dt*dolfin.inner(dolfin.grad(self.u), dolfin.grad(self.alpha*self.v))*dolfin.dx
        a = dolfin.lhs(F)
        L = dolfin.rhs(F)
        # Solve ice temperature
        dolfin.solve(a==L,self.T,[self.bc_inf,self.bc_iWall])
        # Update previous profile to current
        self.u0.assign(self.T)

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def run(self,verbose=False,initialize_array=True,data_dir=None):
        """
        Iterate the model through the given time array.
        """

        # Initialize outputs
        if initialize_array:
            self.r_result = [np.exp(self.coords[:,0])*self.R_melt]
            self.T_result = [np.array(self.u0.vector()[:]*abs(self.T_inf))]
        for i,t in enumerate(self.ts[1:]):
            if verbose:
                print(round(t*self.t0/60.),end=' min, ')

            # --- Thermal Diffusion --- #
            self.update_boundary_conditions(data_dir=data_dir)
            self.solve_thermal()

            # --- Export --- #
            if t in self.save_times:
                self.r_result = np.append(self.r_result,[np.exp(self.coords[:,0])*self.R_melt],axis=0)
                self.T_result = np.append(self.T_result,[self.u0.vector()[:]*abs(self.T_inf)],axis=0)
