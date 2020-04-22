#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:39:42 2019

@author: benhills
"""

import numpy as np
from analytical_pure_solution import analyticalMelt
from constants_stefan import constantsIceDiver
from concentration_functions import Tf_depression,Hmix
const = constantsIceDiver()
import dolfin

class cylindrical_stefan():
    def __init__(self,const=const):
        """
        Initial Variables
        """

        # Temperature Variables
        self.T_inf = -15.                       # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature
        self.Tf_reg = 1.0
        self.Q_wall = 0.0                       # Heat Source after Melting (W)
        self.Q_initialize = 2500.               # Heat Source for Melting (W)
        self.Q_center = 0.0                     # line source at borehole center (W/m?? TODO)
        self.Q_sol = 0.0

        # Concentration Variables
        self.source_timing = np.inf
        self.source_duration = np.inf
        self.C_init = 0.0                       # Initial Solute Concentration (before injection)
        self.source_mass_final = 1.            # Total Injection Mass
        self.mol_diff = 0.84e-9

        # Domain Specifications
        self.R_center = 0.001                     # Innner Domain Edge
        self.R_inf = 1.                          # Outer Domain Edge
        self.R_melt = 0.04                          # Melt-Out Radius
        self.n = 100
        self.rs = np.linspace(self.R_center,self.R_inf,self.n)
        self.dt = 10.
        self.t_final = 2.*3600.

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
        self.Tstar = self.T_inf/abs(self.T_inf)
        self.Rstar = self.R_melt/self.R_melt
        self.Rstar_center = self.R_center/self.R_melt
        self.Rstar_inf = self.R_inf/self.R_melt
        self.Qstar = self.Q_wall/(2.*np.pi*const.ki*abs(self.T_inf))
        Lv = const.L*const.rhoi     # Latent heat of fusion per unit volume
        self.astar_i = Lv/(const.rhoi*const.ci*abs(self.T_inf))
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_melt**2.

        # Dimensionless Constants
        self.St = const.ci*(self.Tf-self.T_inf)/const.L
        self.Lewis = const.ki/(const.rhoi*const.ci*self.mol_diff)

        # Tranform to a logarithmic coordinate system so that there are more points near the borehole wall.
        self.w0 = np.log(self.Rstar)
        self.wf = np.log(self.Rstar_inf)
        self.w_center = np.log(self.Rstar_center)
        self.ws = np.log(self.rs)

        self.flags.append('log_transform')

    def get_domain(self):
        """
        Define the Finite Element domain for the problem
        """

        # Finite Element Mesh in solid
        self.ice_mesh = dolfin.IntervalMesh(self.n,self.w0,self.wf)
        self.ice_V = dolfin.FunctionSpace(self.ice_mesh,'CG',1)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates().copy()
        self.ice_idx_wall = np.argmin(self.ice_coords)

        self.flags.append('get_domain')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def get_initial_conditions(self):
        """
        Set the initial condition at the end of melting (melting can be solved analytically
        """

        # --- Initial states --- #
        # (ice temperature)
        self.u0_i = dolfin.Function(self.ice_V)
        T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.ice_coords[:,0])*self.R_melt,self.T_inf,self.Q_initialize,R_target=self.R_melt)
        self.u0_i.vector()[:] = T/abs(self.T_inf)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(self.t_melt,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0
        # Define the ethanol source
        self.source_timing = self.ts[np.argmin(abs(self.ts-self.source_timing/self.t0))]
        if 'gaussian_source' in self.flags:
            self.source_duration /= self.t0
            self.source = self.source_mass_final/(self.source_duration*np.sqrt(np.pi))*np.exp(-((self.ts-self.source_timing)/self.source_duration)**2.)
        else:
            self.source = np.zeros_like(self.ts)
            self.source[np.argmin(abs(self.ts-self.source_timing))] = self.source_mass_final/self.dt

        # --- Define the test and trial functions --- #
        self.u_i = dolfin.TrialFunction(self.ice_V)
        self.v_i = dolfin.TestFunction(self.ice_V)
        self.T_i = dolfin.Function(self.ice_V)
        self.C = self.C_init
        self.Tf_wall = Tf_depression(self.C)
        # Get the updated solution properties
        self.rhos = self.C + const.rhow*(1.-self.C/const.rhoe)
        self.cs = (self.C/const.rhoe)*const.ce+(1.-(self.C/const.rhoe))*const.cw
        self.ks = (self.C/const.rhoe)*const.ke+(1.-(self.C/const.rhoe))*const.kw
        self.rhos_wall,self.cs_wall,self.ks_wall = self.rhos,self.cs,self.ks

        self.flags.append('get_ic')

    def get_boundary_conditions(mod):
        """
        Define Boundary Conditions
        """

        # Left boundary is the center of the borehole, so it is at the melting temperature
        class iWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.ice_coords[mod.ice_idx_wall] + const.tol
        # Right boundary is the far-field temperature
        class Inf(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                    return on_boundary and x[0] > mod.wf - const.tol

        # Initialize boundary classes
        mod.iWall = iWall()
        mod.Inf = Inf()
        # Set the Dirichlet Boundary condition at
        mod.bc_inf = dolfin.DirichletBC(mod.ice_V, mod.Tstar, mod.Inf)

        mod.flags.append('get_bc')


    def initiate_solution_diffusion(mod):

        # --- Get new domain --- #
        mod.sol_mesh = dolfin.IntervalMesh(mod.n,mod.Rstar_center,mod.Rstar)
        mod.sol_V = dolfin.FunctionSpace(mod.sol_mesh,'CG',1)
        mod.sol_mesh.coordinates()[:] = np.log(mod.sol_mesh.coordinates())
        mod.sol_coords = mod.sol_V.tabulate_dof_coordinates().copy()
        mod.sol_idx_wall = np.argmax(mod.sol_coords)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        mod.ts = np.arange(mod.t_init,mod.t_final+mod.dt,mod.dt)/mod.t0
        mod.dt /= mod.t0
        # Define the ethanol source
        mod.source_timing = mod.source_timing/mod.t0
        mod.source_duration /= mod.t0
        mod.source = mod.source_mass_final/(mod.source_duration*np.sqrt(np.pi))*np.exp(-((mod.ts-mod.source_timing)/mod.source_duration)**2.)

        # --- Set Initial Conditions --- #
        # (solution temperature),
        mod.u0_s = dolfin.Function(mod.sol_V)
        mod.u0_s.vector()[:] = mod.Tf_wall
        # (solution concentration),
        mod.u0_c = dolfin.project(dolfin.Constant(mod.C),mod.sol_V)

        # --- Set up the variational Problem --- #
        mod.u_s = dolfin.TrialFunction(mod.sol_V)
        mod.v_s = dolfin.TestFunction(mod.sol_V)
        mod.T_s = dolfin.Function(mod.sol_V)
        mod.C = dolfin.project(dolfin.Constant(mod.C),mod.sol_V)
        mod.Tf = Tf_depression(mod.C)/abs(mod.T_inf)
        mod.Tf_wall = dolfin.project(mod.Tf,mod.sol_V).vector()[mod.sol_idx_wall]

        # --- Get the solution properties --- #
        mod.rhos = dolfin.project(dolfin.Expression('C + rhow*(1.-C/rhoe)',degree=1,C=mod.C,rhow=const.rhow,rhoe=const.rhoe),mod.sol_V)
        mod.cs = dolfin.project(dolfin.Expression('ce*(C/rhoe) + cw*(1.-C/rhoe)',degree=1,C=mod.C,cw=const.cw,ce=const.ce,rhoe=const.rhoe),mod.sol_V)
        mod.ks = dolfin.project(dolfin.Expression('ke*(C/rhoe) + kw*(1.-C/rhoe)',degree=1,C=mod.C,kw=const.kw,ke=const.ke,rhoe=const.rhoe),mod.sol_V)
        mod.rhos_wall = mod.rhos.vector()[mod.sol_idx_wall]
        mod.cs_wall = mod.cs.vector()[mod.sol_idx_wall]
        mod.ks_wall = mod.ks.vector()[mod.sol_idx_wall]

        # --- Boundary Conditions --- #
        # Liquid boundary condition at hole wall (same temperature as ice)
        class sWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > mod.sol_coords[mod.sol_idx_wall] - const.tol
        # center flux
        class center(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.w_center + const.tol

        # Initialize boundary classes
        mod.sWall = sWall()
        mod.center = center()
        # This will be used in the boundary condition for mass diffusion
        mod.boundaries = dolfin.MeshFunction("size_t", mod.sol_mesh, 0) # this index 0 is an alternative to the command boundaries.set_all(0)
        mod.sWall.mark(mod.boundaries, 1)
        mod.center.mark(mod.boundaries, 2)
        mod.sds = dolfin.Measure("ds")(subdomain_data=mod.boundaries)


    # ----------------------------------------------------------------------------------------------------------------------------------------

    def update_boundary_conditions(self):
        """
        Update the boundary conditions for new wall location and new wall temperature.
        """

        # Update the thermal boundary condition at the wall based on the current concentration
        # update the melting temperature
        self.Tf = Tf_depression(self.C)/abs(self.T_inf)
        self.Tf_wall = self.Tf
        # Reset ice boundary condition
        self.bc_iWall = dolfin.DirichletBC(self.ice_V, self.Tf_wall, self.iWall)

    def solve_molecular(self):
        """
        Solve the molecular diffusion problem
        if 'solve_sol_mol' is not in the flags list this will be an instantaneous mixing problem.
        """
        # Instantaneous Mixing
        # Recalculate the solution concentration after wall moves
        self.C *= (self.Rstar/np.exp(self.ice_coords[self.ice_idx_wall,0]))**2.
        # Recalculate the freezing temperature
        self.Tf_last = self.Tf
        self.Tf = Tf_depression(self.C)
        # Get the updated solution properties
        self.rhos = self.C + const.rhow*(1.-self.C/const.rhoe)
        self.cs = (self.C/const.rhoe)*const.ce+(1.-(self.C/const.rhoe))*const.cw
        self.ks = (self.C/const.rhoe)*const.ke+(1.-(self.C/const.rhoe))*const.kw
        self.rhos_wall,self.cs_wall,self.ks_wall = self.rhos,self.cs,self.ks

    def solve_thermal(self):
        """
        Solve the thermal diffusion problem.
        Both ice and solution.
        """

        ### Solve heat equation
        alphalog_i = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',degree=1,astar=self.astar_i),self.ice_V)
        # Set up the variational form for the current mesh location
        F_i = (self.u_i-self.u0_i)*self.v_i*dolfin.dx + self.dt*dolfin.inner(dolfin.grad(self.u_i), dolfin.grad(alphalog_i*self.v_i))*dolfin.dx
        a_i = dolfin.lhs(F_i)
        L_i = dolfin.rhs(F_i)
        # Solve ice temperature
        dolfin.solve(a_i==L_i,self.T_i,[self.bc_inf,self.bc_iWall])
        # Update previous profile to current
        self.u0_i.assign(self.T_i)

    def injection_energy_balance(self,source):
        """
        At the time of injection assume that melting happens all at once
        """

        # Calculate the injection source actually adds to total concentration
        C_inject = source*self.dt/(np.pi*(self.Rstar*self.R_melt)**2.)
        # Hard set on concentration (assume that it mixes quickly)
        self.C += C_inject

        # enthalpy of mixing, always exothermic so gives off energy (J m-3)
        # put this added energy toward uniformly warming the solution
        H,phi = Hmix(C_inject)
        #phi_dT = -phi/(self.rhos*self.cs)/abs(self.T_inf)
        phi_dT = -phi/(const.rhow*const.cw)/abs(self.T_inf)

        # Hard set on solution temperature (assume that the mixing energy spreads evenly)
        self.Tf = Tf_depression(self.C)/abs(self.T_inf)
        # Bump the last freezing temperature up
        # this way the mixing enthalpy is accounted for in wall movement
        self.Tf_last = (Tf_depression(self.C-C_inject)+phi_dT)/abs(self.T_inf)

    def move_wall(self,const=const):
        """
        Melting/Freezing at the hole wall
        This is the Stefan condition.
        """

        # --- Calculate Distance Wall Moves --- #

        # Melting/freezing at the hole wall from prescribed flux and temperature gradient
        # Humphrey and Echelmeyer (1990) eq. 13
        dRdt = dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_i.dx(0),self.ice_V).vector()[self.ice_idx_wall] + \
                    self.Qstar/self.Rstar
        # calculate sensible heat contribution toward wall melting associated with change in the freezing temp
        dRdt += np.sqrt(((self.rhos_wall*self.cs_wall*(self.Tf_last-self.Tf)*self.Rstar**2.)/(const.rhow*const.L))+self.Rstar**2.)-self.Rstar
        self.dR = dRdt*self.dt

        # Is the hole completely frozen? If so, exit
        Frozen = np.exp(self.ice_coords[self.ice_idx_wall,0])+self.dR < 0.
        if Frozen:
            self.flags.append('Frozen')
            return

        # --- Move the Mesh --- ###

        # stretch mesh rather than uniform displacement
        dRsi = self.dR/(self.Rstar_inf-self.Rstar)*(self.Rstar_inf-np.exp(self.ice_coords[:,0]))
        # Interpolate the points onto what will be the new mesh (ice)
        self.ice_idx_extrapolate = np.exp(self.ice_coords[:,0]) + self.dR >= self.Rstar
        u0_i_hold = self.u0_i.vector()[:].copy()
        u0_i_hold[self.ice_idx_extrapolate] = np.array([self.u0_i(xi) for xi in np.log(np.exp(self.ice_coords[self.ice_idx_extrapolate,0])+dRsi[self.ice_idx_extrapolate])])
        u0_i_hold[~self.ice_idx_extrapolate] = self.Tf_wall
        self.u0_i.vector()[:] = u0_i_hold[:]
        # advect the mesh according to the movement of the hole wall
        dolfin.ALE.move(self.ice_mesh,dolfin.Expression('std::log(exp(x[0])+dRsi*(Rstar_inf-exp(x[0])))-x[0]',degree=1,dRsi=self.dR/(self.Rstar_inf-self.Rstar),Rstar_inf=self.Rstar_inf))
        self.ice_mesh.bounding_box_tree().build(self.ice_mesh)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates().copy()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def conservation(self,const=const):
        x_exp = dolfin.Expression('pow(exp(x[0]),2)',degree=1)
        self.EnCon = dolfin.assemble(self.cs*self.rhos*x_exp*self.u0_s*dolfin.dx)+\
                        const.ci*const.rhoi*dolfin.assemble(x_exp*self.u0_i*dolfin.dx)
        self.MassCon = dolfin.assemble(x_exp*self.u0_c*dolfin.dx)
        return

    def run(self,verbose=False,initialize_array=True):
        """
        Iterate the model through the given time array.
        """

        if initialize_array:
            self.r_ice_result = [np.exp(self.ice_coords[:,0])*self.R_melt]
            self.T_ice_result = [np.array(self.u0_i.vector()[:]*abs(self.T_inf))]
        for i,t in enumerate(self.ts[1:]):
            if verbose:
                print(round(t*self.t0/60.),end=' min, ')

            # --- Ethanol Injection --- #
            self.injection_energy_balance(self.source[i+1])

            # --- Thermal Diffusion --- #
            self.update_boundary_conditions()
            self.solve_thermal()

            # --- Move the Mesh --- #
            self.move_wall()
            # break the loop if the hole is completely frozen
            if 'Frozen' in self.flags:
                print('Frozen Hole!')
                break

            # --- Molecular Diffusion --- #
            self.solve_molecular()

            # Save the new wall location
            self.Rstar = np.exp(self.ice_coords[self.ice_idx_wall,0])

            # --- Export --- #
            if t in self.save_times:
                self.r_ice_result = np.append(self.r_ice_result,[np.exp(self.ice_coords[:,0])*self.R_melt],axis=0)
                self.T_ice_result = np.append(self.T_ice_result,[self.u0_i.vector()[:]*abs(self.T_inf)],axis=0)