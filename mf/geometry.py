import gmsh
import numpy as np
from mf.fem import gmsh2dolfin, write_gmsh

def CircularMesh(path, mesh_name, R, h, plot_info):
    # Create membrane mesh with Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(mesh_name)
    factory = gmsh.model.geo
    
    # Create geometry
    # Add Points, px, py, pz, resolution, tag 
    factory.addPoint(-R,  0 , 0, h, 1)                          
    factory.addPoint(0,   0 , 0, h, 2)
    factory.addPoint(R,   0 , 0, h, 3)
    
    factory.addPoint(0,  -R, 0, h, 4)
    factory.addPoint(0,   R, 0, h, 5)
    
    # Add arcs, p1, center, p2, tag
    factory.addCircleArc(3, 2, 4, 10)                            
    factory.addCircleArc(4, 2, 1, 11)
    
    factory.addCircleArc(1, 2, 5, 12)
    factory.addCircleArc(5, 2, 3, 13)
    
    # Add curve loop (closes the boundary) [tag_line1,.. tag_lineN], tag 
    id_border = factory.addCurveLoop([10, 11, 12, 13])                  
    
    # create a planar surface with tag = id_surface
    id_surface = factory.addPlaneSurface([id_border])                  
    
    # sync changes
    factory.synchronize()                                       
    # create mesh
    gmsh.model.mesh.generate(2)                                 
    
    # add a group name to the surface
    surface_string_tag = 'my_surface'
    tag_surface = gmsh.model.addPhysicalGroup(2, [id_surface])    
    # set the name to element in this sirface, (dim, tag, name)
    gmsh.model.setPhysicalName(2, tag_surface, surface_string_tag)        
    
    # add a group name to the boundary
    bord_string_tag = 'my_bords'
    tag_bords  = gmsh.model.addPhysicalGroup(1, [10, 11, 12, 13])                    # REMEMBER THIS TAG FOR BC
    # set the name to element in this sirface, (dim, tag, name)
    gmsh.model.setPhysicalName(1, tag_bords, bord_string_tag)                  # dim, tag, name

    
    # Comment/Uncomment to see the mesh in gmsh
    if plot_info == 'plot':
        gmsh.fltk.run()                                             
    # Write mesh
    write_gmsh(path,mesh_name)                                       
    # Don't forget to finalize gmsh
    gmsh.finalize()  

    fmesh, mf_boundary = gmsh2dolfin(path, mesh_name, '2D', bord_string_tag, surface_string_tag)            

    return fmesh, mf_boundary, tag_bords

def RectangularMesh(path, mesh_name, Lx,Ly, h, plot_info):
    # Create membrane mesh with Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(mesh_name)
    factory = gmsh.model.geo
    
    # Create geometry
    # Add Points, px, py, pz, resolution, tag 
    factory.addPoint(0,   0,  0, h, 1)                          
    factory.addPoint(Lx,  0,  0, h, 2)
    factory.addPoint(Lx,  Ly, 0, h, 3)
    factory.addPoint(0,   Ly, 0, h, 4)
    
    # Add arcs, p1, center, p2, tag
    factory.addLine(1, 2, 10)
    factory.addLine(2, 3, 11)
    factory.addLine(3, 4, 12)
    factory.addLine(4, 1, 13)

    # Add curve loop (closes the boundary) [tag_line1,.. tag_lineN], tag 
    id_border = factory.addCurveLoop([10, 11, 12, 13])                  
    
    # create a planar surface with tag = id_surface
    id_surface = factory.addPlaneSurface([id_border])                  
    
    # sync changes
    factory.synchronize()                                       
    # create mesh
    gmsh.model.mesh.generate(2)                                 
    
    # add a group name to the surface
    surface_string_tag = 'my_surface'
    tag_surface = gmsh.model.addPhysicalGroup(2, [id_surface])    
    # set the name to element in this sirface, (dim, tag, name)
    gmsh.model.setPhysicalName(2, tag_surface, surface_string_tag)        
    
    # add a group name to the boundary
    bord_string_tag = 'my_bords'
    tag_bords  = gmsh.model.addPhysicalGroup(1, [10, 11, 12, 13])                    # REMEMBER THIS TAG FOR BC
    # set the name to element in this sirface, (dim, tag, name)
    gmsh.model.setPhysicalName(1, tag_bords, bord_string_tag)                  # dim, tag, name

    
    # Comment/Uncomment to see the mesh in gmsh
    if plot_info == 'plot':
        gmsh.fltk.run()                                             
    # Write mesh
    write_gmsh(path,mesh_name)                                                                                                                  
    # Don't forget to finalize gmsh
    gmsh.finalize()  

    fmesh, mf_boundary = gmsh2dolfin(path, mesh_name, '2D', bord_string_tag, surface_string_tag)            

    return fmesh, mf_boundary, tag_bords

def PolygonalMesh(path, mesh_name, N, a, h1, plot_info):
    
    alpha = np.deg2rad(360/N)
    phi = alpha/2+np.pi/2
    beta = np.deg2rad(90) - alpha/2
    b = np.sqrt( np.pi*a**2* np.sin(alpha/2) / (N * np.sin(beta)))
    
    l = 2*b*np.sin(beta)/np.sin(alpha/2)
    area_t = 1/2*b*N*l
    
    # Create mesh
    import gmsh, sys
    from mf.fem import gmsh2dolfin, write_gmsh
    # Mesh generation with GMSH
    gmsh.initialize(sys.argv)
    
    # Ask GMSH to display information in the terminal
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(mesh_name)
    
    model = gmsh.model
    model.add("MyPlate")
    
    R = np.sqrt(b**2+ (l/2)**2)
    
    # Create Polygon's coordinates
    vertex, borders, ptA, ptB = ( [] for i in range(4))
    
    for jj in range(N):
        vertex.append(model.geo.addPoint( R*np.cos(2*np.pi*jj/N - phi), R*np.sin(2*np.pi*jj/N- phi), 0, h1))
    
    # Create Point for the center of the circle
    center = model.geo.addPoint(0,0,0, h1)
    # Create 3 Points on the circle
    n_arc = 3
    points = []
    for j in range(n_arc):
      points.append(model.geo.addPoint(a*np.cos(2*np.pi*j/n_arc), a*np.sin(2*np.pi*j/n_arc), 0, h1))
    
    # Create 3 circle arc
    borders2 = []
    for j in range(n_arc):
      borders2.append(model.geo.addCircleArc(points[j],center,points[(j+1)%n_arc]))
    
    for j in range(N):
        if j < N:
            borders.append(gmsh.model.geo.addLine(vertex[j-1],vertex[j]))
    
    # Curveloop and Surface
    curveloop = model.geo.addCurveLoop(borders)
    id_surface = model.geo.addPlaneSurface([curveloop])
    
    # This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
    gmsh.model.geo.synchronize()
    
    # set algorithm "Packing of parallelograms" (experimental =9)
    gmsh.model.mesh.setAlgorithm(2, id_surface, 9)
    gmsh.option.setNumber('Mesh.MeshSizeFactor', h1)
    gmsh.model.mesh.generate(2)                                 # 2D mesh
    
    # Bord phisical group
    bord_string_tag = "my_bords"
    tag_bords = gmsh.model.addPhysicalGroup(1, borders)         
    gmsh.model.setPhysicalName(1, tag_bords, bord_string_tag)   # dim, tag, name
    
    # Surface phisical group
    surface_string_tag = "my_surface"
    tag_dom = gmsh.model.addPhysicalGroup(2, [id_surface])       # Delete original tag when fragment
    gmsh.model.setPhysicalName(2, tag_dom, surface_string_tag)   # dim, tag, name
    
   # Comment/Uncomment to see the mesh in gmsh
    if plot_info == 'plot':
        gmsh.fltk.run()                                             
    # Write mesh
    write_gmsh(path,mesh_name)                                       
    # Don't forget to finalize gmsh
    gmsh.finalize()  

    fmesh, mf_boundary = gmsh2dolfin(path, mesh_name, '2D', bord_string_tag, surface_string_tag)            

    return fmesh, mf_boundary, tag_bords

def EllipticalMesh(path, mesh_name, Lx, Ly, h1, plot_info):
    # Create mesh
    import gmsh, sys
    from mf.fem import gmsh2dolfin, write_gmsh
    # Mesh generation with GMSH
    gmsh.initialize(sys.argv)
    
    # Ask GMSH to display information in the terminal
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(mesh_name)
    factory = gmsh.model.occ
    model = gmsh.model
    model.add("MyPlate")
    
    borders = factory.addEllipse(0,0,0,Lx/2, Ly/2)
    
    # Curveloop and Surface
    curveloop = factory.addCurveLoop([borders])
    id_surface = factory.addPlaneSurface([curveloop])
    
    # sync changes
    factory.synchronize()                                       
    
    # set algorithm "Packing of parallelograms" (experimental =9)
    gmsh.model.mesh.setAlgorithm(2, id_surface, 9) # 8 creates the same mesh
    gmsh.option.setNumber('Mesh.MeshSizeFactor', h1)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h1/5)
    gmsh.model.mesh.generate(2)                                 # 2D mesh
    
    # Bord phisical group
    bord_string_tag = "my_bords"
    tag_bords = gmsh.model.addPhysicalGroup(1, [borders])         
    gmsh.model.setPhysicalName(1, tag_bords, bord_string_tag)   # dim, tag, name

    # Surface phisical group
    surface_string_tag = "my_surface"
    tag_dom = gmsh.model.addPhysicalGroup(2, [id_surface])       # Delete original tag when fragment
    gmsh.model.setPhysicalName(2, tag_dom, surface_string_tag)   # dim, tag, name
    
   # Comment/Uncomment to see the mesh in gmsh
    if plot_info == 'plot':
        gmsh.fltk.run()                                             
    # Write mesh
    #write_gmsh(path,mesh_name)                                       
    gmsh.write(path+mesh_name)                                       
    # Don't forget to finalize gmsh
    gmsh.finalize()  

    fmesh, mf_boundary = gmsh2dolfin(path, mesh_name, '2D', bord_string_tag, surface_string_tag)            

    return fmesh, mf_boundary, tag_bords