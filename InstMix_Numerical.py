#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:10:04 2018

@author: benhills
"""


import numpy as np
from dolfin import *
parameters['allow_extrapolation'] = True

from Constants import *
const = constantsIceDiver()

#--------------------------------------------------------------------------------------------------------------------------------

class cylindricalInstMix():

    def __init__(self):
        # Problem variables
        Tinf = -5.                             # Far Field Temperature
        #R0 = 0.01                               # Initial Radius
        R0 = 0.04
        Q, Qmelt = 0., 2500.                 # Heat flux for problem and for melting (Qmelt is used in the analytical solution)

        # Nondimensionalize
        Tstar = Tinf/abs(Tinf)
        Rstar = R0/R0
        Qstar = Q/(2.*np.pi*const.ki*abs(Tinf))
        kstar = const.L*const.rhoi/(const.rhoi*const.ci*abs(Tinf))
        t0 = const.rhoi*const.ci/const.ki*kstar*R0**2.

        # industrial solvents handbook, percent by mass
        Tfd = np.load('../EthanolProperties/FreezingDepression_PBM.npy')
        # linear interpolation between points
        from scipy.interpolate import interp1d
        Tfd_interp = interp1d(Tfd[0], Tfd[1])

        # Times
        t_inject = 2.*3600.
        t_final = 2.*3600.

        #V_inject = .00112551                         # injection volume (set to get R^2 of 0.04 m)

        #for t_inject in np.linspace(0,7.5*3600.,10):

        # Reinitialize
        Rstar = 1.
        pbv,pbv_last = 0.,0.

        Tinit

    #--------------------------------------------------------------------------------------------------------------------------------

    def rhoe(T,A=99.39,B=0.31,C=513.18,D=0.305):
        T = T+const.Tf0
        return A/(B**(1+(1-T/C)**D))

    def pbmPBV(pbv,T=Tinf):
        # calculate the density of the solution
        rho_s = const.rhow*(1.-pbv)+rhoe(T)*pbv
        # calculate the percent by mass of the solution
        pbm = pbv*(rhoe(T)/rho_s)
        return pbm

    def Tf(pbv,Tfd=Tfd,Tfd_interp=Tfd_interp):
        # freezing point depression
        if pbv < 0.:
            return Tfd_interp(0.)
        elif pbv > 1.:
            return Tfd_interp(1.)
        else:
            # convert to percent by mass
            pbm = pbmPBV(pbv)
            return Tfd_interp(pbm)

    def Tf_optimize(pbv,Tcurrent):
        return Tf(pbv) - Tcurrent

    # Enthalpy of mixing
    def Hmix(pbv,T=0.,mmass_e=46.07,mmass_w=18.02):
        if pbv > 1.:
            pbv = 1.
        if pbv < 0.:
            pbv = 0.
        # calculate the density of the solution
        rho_s = const.rhow*(1-pbv)+rhoe(T)*pbv
        # calculate the percent by mass of the solution
        pbm = pbv*(rhoe(T)/rho_s)
        # mole fraction
        Xa = pbm*mmass_e/((1.-pbm)*mmass_w+pbm*mmass_e)
        Xw = 1.-Xa
        # Energy
        return 1000.*(-10.6*Xw**6.*Xa-1.2*Xw*Xa+.1*Xw*Xa**2.)*(1000./mmass_e)

    # Melting/Freezing at the hole wall
    def moveWall(Mesh,Rstar,pbv,pbv_last,dt,T,Qstar,Tinf=Tinf,MIX=0.):
        # update the melting temperature
        Tm = Tf(pbv)
        Tm_last = Tf(pbv_last)
        # heat capacity of the solution
        cs = pbv*const.ce+(1-pbv)*const.cw
        rhos = pbv*rhoe(Tinf)+(1-pbv)*const.rhow
        # melting at hole wall from sensible heat in the solution
        dR = np.sqrt(((rhos*cs*(Tm_last-Tm)*Rstar**2.)/(const.rhow*const.L))+Rstar**2.) - Rstar
        # melting/freezing at the hole wall from prescribed flux and temperature gradient
        # Humphrey and Echelmeyer (1990) eq. 13
        dR = dt*(project(Expression('exp(-x[0])')*T.dx(0),V).vector()[0] + Qstar/Rstar)
        # melting at the hole wall from enthalpy of mixing
        if MIX != 0.:
            dR += np.sqrt(((rhos*(-MIX)*Rstar**2.)/(const.rhow*const.L))+Rstar**2.) - Rstar
        # Is the hole completely frozen?
        Frozen = np.exp(coords[:,0])[0]+dR[0] < 0.
        if Frozen:
            return Mesh,Rstar,pbv,pbv_last,None,Frozen
        # Interpolate the points onto what will be the new mesh
        u0.vector()[:] = np.array([u0(xi) for xi in np.log(np.exp(coords[:,0])+dR[0])])
        # advect the mesh according to the movement of teh hole wall
        ALE.move(Mesh,Expression('log(exp(x[0])+dR)-x[0]',dR=dR[0]))
        Mesh.bounding_box_tree().build(Mesh)
        # Reset boundary condition
        bc_c = DirichletBC(V, Tm/abs(Tinf), Center())
        bcs = [bc_c,bc_inf]
        # update the concentration of solution in the hole
        pbv_last = pbv
        pbv = pbv_last*(Rstar/np.exp(coords[0][0]))**2.
        Rstar = np.exp(coords[0][0])
        return Mesh,Rstar,pbv,pbv_last,bcs,Frozen

    #--------------------------------------------------------------------------------------------------------------------------------

    ### Define the domain for the problem ###

    def drawMesh(self):
        # Finite Element Mesh
        w0,wf,n = np.log(Rstar),np.log(100.*Rstar), 100
        Mesh = IntervalMesh(n,w0,wf)
        coords = Mesh.coordinates()
        V = FunctionSpace(Mesh,'CG',1)

    #--------------------------------------------------------------------------------------------------------------------------------

    def setIC(self):

        ### Define Initial Condition ###
        u0 = Function(V)
        u0.vector()[:] = Tinit

        dt = 60.
        ts = np.arange(t_drill,t_final+dt,dt)
        t_inject = ts[np.argmin(abs(ts-t_inject))]
        ts /= t0
        dt /= t0


    def setBC(self):
        ### Define Boundary Conditions ###

        # Left boundary is the center of the borehole, so it is at the melting temperature
        class Center(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < coords[0] + const.tol
        bc_c = DirichletBC(V, Tf(0.), Center())

        # Right boundary is the ambient temperature of the ice
        class Inf(SubDomain):
            def inside(self, x, on_boundary):
                    return on_boundary and x[0] > wf - tol
        bc_inf = DirichletBC(V, Tstar, Inf())

        bcs = [bc_c,bc_inf]

#--------------------------------------------------------------------------------------------------------------------------------

    def main(self):
        ### Define the variational problem ###

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        T = Function(V)

        ### Iterate ###

        # Output arrays
        R_out = np.array([])
        pbv_out = np.array([0])
        T_out = np.array([u0.vector()[0]])
        G_out = np.array(abs(T0)/R0*project(Expression('exp(-x[0])')*u0.dx(0),V).vector()[0])
        from scipy.optimize import fsolve

        for t in ts[1:]:
            # Write output
            #R_out = np.append(R_out,Rstar*R0)
            #pbv_out = np.append(pbv_out,pbv)
            print 'Injection:', round(t_inject/3600.,1), '     V:', V_inject, '    Current time:', round(t*t0/3600.,3), 'hr'
            # Update the variational form for the current mesh location
            kappa = project(Expression('kstar*exp(-2.*x[0])',kstar=kstar),V)
            F = (u-u0)*v*dx + dt*inner(grad(u), grad(dot(kappa,v)))*dx
            a = lhs(F)
            L = rhs(F)

            # Melt/Freeze wall, calculate the melting temperature, and reset the boundary condition
            #if t*t0 > t_drill:
                #Qstar = 0.
            #if abs(t*t0 - t_inject) < const.tol:
            #    print 'Inject!'
            #    pbv = V_inject/(np.pi*(Rstar*R0)**2.)
            #    Mesh,Rstar,pbv,pbv_last,bcs,Frozen = moveWall(Mesh,Rstar,pbv,pbv_last,dt,u0,Qstar,MIX=Hmix(pbv))
            #else:
            #    Mesh,Rstar,pbv,pbv_last,bcs,Frozen = moveWall(Mesh,Rstar,pbv,pbv_last,dt,u0,Qstar)\


            if Frozen:
                print 'Frozen'
                break

            # Solve
            solve(a==L,T,bcs)
            u0.assign(T)


            pbv = fsolve(Tf_optimize,.5,args=(T.vector()[0]*abs(Tinf)))

            G_out = np.append(G_out,abs(Tinf)/R0*project(Expression('exp(-x[0])')*T.dx(0),V).vector()[0])
            #T_out = np.append(T_out,T.vector()[0]*abs(Tinf))
            #pbv_out = np.append(pbv_out,pbv)
