import gmsh
from mf.settings import clean_env
import numpy as np
clean_env()

# Create piston with Gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
mesh_name = "piston.msh"
gmsh.model.add(mesh_name)
factory = gmsh.model.geo

cm = 1e-02
R1 = 25 * cm

h = 0.04

# Create geometry
# Add Points
factory.addPoint(-R1, 0 , 0, h, 1)                          # px, py, pz, resolution, tag 
factory.addPoint(0,   0 , 0, h, 2)
factory.addPoint(R1,  0 , 0, h, 3)

factory.addPoint(0,  -R1, 0, h, 4)
factory.addPoint(0,   R1, 0, h, 5)

# Add arcs
factory.addCircleArc(3, 2, 4, 10)                           # p1, center, p2, tag 
factory.addCircleArc(4, 2, 1, 11)

factory.addCircleArc(1, 2, 5, 12)
factory.addCircleArc(5, 2, 3, 13)

# Add curve loop (closes the boundary)
factory.addCurveLoop([10, 11, 12, 13], 20)                  # [tag_line1,.. tag_lineN], tag 

id_surface = factory.addPlaneSurface([20])                  # create a planar surface with tag = id_

factory.synchronize()                                       # sync changes
gmsh.model.mesh.generate(2)                                 # create mesh

# # # surface phisical group
tag_piston= gmsh.model.addPhysicalGroup(2, [id_surface])    # add a group name to the surface
gmsh.model.setPhysicalName(2, tag_piston, "piston")         # set the name to element in this sirface, (dim, tag, name)

# # # =============================================================================
# # # the command line arguments:
# if '-nopopup' not in sys.argv:
#gmsh.fltk.run()                                             # Uncomment to see the mesh in gmsh
# # # =============================================================================
gmsh.write(mesh_name)                                       # Write mesh
gmsh.finalize()                                             # Dont forget to finalize gmsh

#%% bempp
import bempp.api

# Define Wavenumber and Acoustic Properties
k = 50
c = 343
a = R1
rho = 1.21
Wdot = 1
omega = c*k
lmda = 2*np.pi*c/omega
print('a/lambda= %.2f' % (a/lmda)) 

# Create grid from mesh (by meshio call in bempp)
grid0 = bempp.api.grid.io.import_grid(mesh_name)                                # domain_indices=tag_piston implicit in gmsh

#%% Define piecewise constant basis functions
pc_space = bempp.api.function_space(grid0, "DP", 0)

#Define velocity boundary condition
@bempp.api.complex_callable
def piston_normal_velocity(x, n, domain_index, result):
    result[0] = Wdot

vn = bempp.api.GridFunction( pc_space, fun=piston_normal_velocity )

#%% Create axes
from matplotlib import pyplot as plt
import meshio
from mf.plots import PlotSettings, ColorbarSettings
from scipy.spatial import Delaunay
my_map = plt.get_cmap('seismic')

fig = plt.figure(figsize =  (10,8))
my_GridSpec = [2,3]
grid_ = plt.GridSpec(my_GridSpec[0], my_GridSpec[1], wspace=0.4, hspace=0.3)
ax1 = plt.subplot(grid_[0, 0], projection='3d')
ax2 = plt.subplot(grid_[0, 1])
# ax3 = plt.subplot(grid_[0, 2], projection='polar')
ax4 = plt.subplot(grid_[1, :3])
PlotSettings(fig,[ax1,ax2,ax4])

# 1. Plot piston grid
plt.sca(ax1)
my_mesh = meshio.read(mesh_name)
coordinates = my_mesh.points[:,0:2]#, my_mesh.points[:,1]]#, my_mesh.points[:,1]
tri = Delaunay(coordinates)
spl = tri.simplices
plt.triplot(coordinates[:,0], coordinates[:,1], tri.simplices)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Piston grid')

#%% 2. Evaluate on field points on the plane (x,y=0, z)
# Define on plane field points
Nx = 200
Nz = 200

xmin, xmax, zmin, zmax = [-5*a, 5*a, 0,15*a]
plot_grid = np.mgrid[xmin:xmax:Nx*1j , zmin:zmax:Nz*1j]
field_point = np.vstack(( plot_grid[0].ravel(),
                          np.zeros(plot_grid[0].size), 
                          plot_grid[1].ravel() ))

# Create potencial with Numba
L_pot = bempp.api.operators.potential.helmholtz.single_layer(pc_space, field_point, k, assembler="dense", device_interface="numba")
# Evaluate potenvial given velocity boundary conditions
phi_field = L_pot.evaluate(vn)
# Compute the pressure field given the potential
p_field_Numba = -1j*rho*omega*phi_field

# Reshape to 2d surface
P_Numba = p_field_Numba.reshape((Nx,Nz))

# Plot surface solution
im2  = ax2.imshow(np.real(P_Numba),  cmap = my_map, extent = [zmin,zmax, xmin,xmax])
ax2.set_xlabel('z')
ax2.set_ylabel('x')
ax2.set_title("On field pressure")
cb1 = plt.colorbar(im2,  ax=ax2)
ColorbarSettings(ax2, cb1, 'pressure')

#%% 4. Evaluate on axis (x=0, y=0, z) field points
# Define on axis field points
z= np.linspace(0,zmax, 500)
field_point = np.vstack((np.zeros(z.size),
                          np.zeros(z.size), 
                          z ))

# Create potencial with Numba
L_pot = bempp.api.operators.potential.helmholtz.single_layer(pc_space, field_point, k, assembler="dense", device_interface="numba")
phi_field = L_pot.evaluate(vn)
p_field_Numba = 2*1j*rho*omega*phi_field                                        # 2 factor because theory of images in Green's function

# Analytical solution
R = np.linalg.norm(field_point, axis=0)                                         # Here R=z
P0 = rho*c*Wdot
p_field_analytical = rho*c*Wdot*( np.exp(1j*k*R) -  np.exp(1j*k*np.sqrt(R**2 + a**2)  ))

# Plot on axis solution
plt1 = ax4.plot(z/a, np.abs(p_field_analytical/P0), label ='Analytical')
plt2 = ax4.plot(z/a, np.abs(p_field_Numba.T/P0), 'r.',markersize=1, label ='bempp')

ax4.set_xlabel('z/a')
ax4.set_ylabel('$|P/P_0|$')
ax4.set_title("On axis pressure")
ax4.legend()


#%% 3. Evaluate Far field directivity (theta)
from  scipy.special  import  jv 
from mf.plots import PolarPlotZoom, set_axis_parameters
# Define angular field points
Rad = a*10                                                                      # Radius of the arc of field points
theta = np.linspace(-np.pi/2, np.pi/2, 200)
field_point = np.vstack((Rad*np.sin(theta),
                        theta*0, 
                        Rad*np.cos(theta) ))

# Create potencial with Numba
L_pot = bempp.api.operators.potential.helmholtz.single_layer(pc_space, field_point, k, assembler="dense", device_interface="numba")
# Evaluate potenvial given velocity boundary conditions
phi_field = L_pot.evaluate(vn)
# Compute the pressure field given the potential
p_field_Numba = -1j*rho*omega*phi_field

# Plot surface solution
p_normalized = np.abs(p_field_Numba.T)/np.max(np.abs((p_field_Numba)))

# Analitical Solution
D = np.abs(2*jv(1,k*a*np.sin(theta))/(k*a*np.sin(theta)))

# Plot Directivity
ax_d1, ax2_d1 = PolarPlotZoom(fig, [my_GridSpec[0], my_GridSpec[1], 3], theta, D, a/lmda, 'fixed', 'Analytical')
ax2_d1.plot(theta*90/(np.pi/2), p_normalized, 'r.', markersize=1, label='bempp')

PlotSettings(fig, ax_d1)
ax_d1.set_title("Direcivity")
ax_d1.legend()

plt.show()















