import meshio
import numpy as np
from fenics import *

def gmsh2dolfin_subd(path, mesh_name, dim, bord_string_tag, surface_string_tag):
    my_mesh = meshio.read(path+mesh_name)
    if dim=='2D':
        set_prune_z = True
    elif dim=='3D':
        set_prune_z = False
    
    def create_mesh(mesh, cell_type, my_tag, prune_z=False):
        cells = np.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
        cell_data = np.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                              for key in mesh.cell_data_dict["gmsh:physical"].keys() if key==cell_type])
    
        # Remove z-coordinates from mesh if we have a 2D cell and all points have the same third coordinate
        points= mesh.points
        if prune_z:
            points = points[:,:2]
        mesh_new = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={my_tag:[cell_data]})
        return mesh_new
    
    border_mesh_name_xdmf = "border_fmv.xdmf"
    dom_mesh_name_xdmf = "domains_fmesh.xdmf"
    
    line_border_mesh = create_mesh(my_mesh, "line", bord_string_tag, prune_z=set_prune_z)
    meshio.write(path + border_mesh_name_xdmf, line_border_mesh)
    
    domains_mesh = create_mesh(my_mesh, "triangle", surface_string_tag, prune_z=set_prune_z)  # Change to false if wanna 2D
    meshio.write(path + dom_mesh_name_xdmf, domains_mesh)
    
    #
    fmesh = Mesh()
    
    with XDMFFile(path + dom_mesh_name_xdmf) as infile:
        infile.read(fmesh)
    
    mvc_border = MeshValueCollection("size_t", fmesh, 1)
    
    with XDMFFile(path + border_mesh_name_xdmf) as infile:
        print("Reading 1d line data into dolfin mvc")
        infile.read(mvc_border, bord_string_tag)
    
    # Load Subdomains
    mvc_domains= MeshValueCollection("size_t", fmesh, 2)
    
    with XDMFFile(path + dom_mesh_name_xdmf) as infile:
        print("Reading 2d line data into subdomains")
        infile.read(mvc_domains, surface_string_tag)
        
    print("Constructing MeshFunction from MeshValueCollection")
    mf_boundary = MeshFunction('size_t',fmesh, mvc_border)
    mf_domains = MeshFunction('size_t',fmesh, mvc_domains)
    
    return fmesh, mf_boundary, mf_domains

def connect_triangles_fem(V, u, mesh, element, plot_info):
    if element == 'dof':
        n = V.dim()                                                     # n nodes
        d = mesh.geometry().dim()                                                        
        dof_coordinates = V.tabulate_dof_coordinates().reshape(n,d)
        xx = dof_coordinates[:,0]
        yy = dof_coordinates[:,1]
        coordinates = dof_coordinates
        
    elif element =='mesh':
        xx = mesh.coordinates()[:,0]
        yy = mesh.coordinates()[:,1]
        coordinates = mesh.coordinates()
        
    spl = mesh.cells()
    cs, se = getCentersTriangles(xx,yy,spl)

    if plot_info=='plot':
        import matplotlib.pyplot as plt
        plt.triplot(xx, yy, spl)
        plt.plot(xx, yy, 'o')
        plt.plot(cs[:,0], cs[:,1], 'rx')

    return cs, se

def getCentersTriangles(xx, yy, spl):
    from scipy.linalg import det
    # get baricenters
    x, y, z= (np.zeros(( len(spl), 3)) for ii in range(0,3))
    for jj in range(0, len(spl)):
        x[jj] = xx[spl[jj,:]]
        y[jj] = yy[spl[jj,:]]

    cx = np.mean(x,1)
    cy = np.mean(y,1)
    cz = 0*cy
    cs = np.c_[cx, cy, cz]

    # get triangle areas
    se = np.zeros( (x.shape[0], ) )
    for jj in range(0, len(se)):
        uv = np.array([ [x[jj][1] - x[jj][0], y[jj][1] - y[jj][0], z[jj][1]-z[jj][0]], \
                        [x[jj][2] - x[jj][0], y[jj][2] - y[jj][0], z[jj][2]-z[jj][0]] ])

        uv0 = np.array([uv[:,1], uv[:,2]])
        uv1 = np.array([uv[:,0], uv[:,2]])
        uv2 = np.array([uv[:,0], uv[:,1]])

        se[jj] = np.sqrt( det(uv0)**2 + det(uv1)**2 + det(uv2)**2 ) /2
        
    return cs, se.reshape(len(cs),1)