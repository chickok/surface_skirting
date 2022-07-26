import glm
import numpy as np
import openmesh as om
import sys

from stl import mesh


# TODO: take in arbitrary mesh (STL) as input via commandline arg
inputMesh = mesh.Mesh.from_file('Part.stl')

#
# STEP ONE: re-orient input mesh so the average of the backsides of its faces is Z-up
# 
# note: this is a naive approach to orientation and will likely not provide a satisfactory
#       position for all input geometries; a better automated solution would require
#       a proper optimization algorithm, e.g. to minimize wall angles
#

# compute average normal of mesh faces
norms = inputMesh.normals
count = 0
avgNorm = [0,0,0]
for n in norms:
    count = count + 1
    avgNorm += n
avgNorm /= count

# construct matrix from rotation between average normal and Z-up vector; transform
# mesh accordingly
avgNormGlm = glm.normalize(glm.vec3(avgNorm[0],avgNorm[1],avgNorm[2]))
zUp = glm.vec3(0,0,1)
theta = glm.acos(glm.dot(avgNormGlm, zUp))
axis = glm.cross(avgNormGlm,zUp)
matGlm = glm.rotate(theta,axis)
matNp = np.array([[matGlm[0][0],matGlm[0][1],matGlm[0][2],matGlm[0][3]],
                  [matGlm[1][0],matGlm[1][1],matGlm[1][2],matGlm[1][3]],
                  [matGlm[2][0],matGlm[2][1],matGlm[2][2],matGlm[2][3]],
                  [matGlm[3][0],matGlm[3][1],matGlm[3][2],matGlm[3][3]]])

inputMesh.transform(matNp)
inputMesh.update_normals()

# get distance of point farthest from XY plane and translate mesh that distance,
# plus some threshold
maxDist = -sys.float_info.max
for p in inputMesh.points:
    if p[2] > maxDist:
        maxDist = p[2]

# arbitrary threshold to provide space between topmost face(s) of mesh and the XY plane
# TODO: expose as user input to determine how much skirt to add
threshold = 5
inputMesh.translate([0,0,-(maxDist + threshold)])

#
# STEP TWO: convert to OpenMesh format for better connectivity information (halfedge data
#           structure allows for easier querying of border edges and faces
#
oMesh = om.TriMesh()
vertDict = dict()
for fv in inputMesh.keys():
    p0 = [fv[0],fv[1],fv[2]]
    p1 = [fv[3],fv[4],fv[5]]
    p2 = [fv[6],fv[7],fv[8]]
    
    p0tup = (fv[0],fv[1],fv[2])
    p1tup = (fv[3],fv[4],fv[5])
    p2tup = (fv[6],fv[7],fv[8])
    
    vh0 = vertDict[p0tup] if p0tup in vertDict else oMesh.add_vertex(p0)
    vh1 = vertDict[p1tup] if p1tup in vertDict else oMesh.add_vertex(p1)
    vh2 = vertDict[p2tup] if p2tup in vertDict else oMesh.add_vertex(p2)
               
    vertDict[p0tup] = vh0
    vertDict[p1tup] = vh1
    vertDict[p2tup] = vh2
    
    oMesh.add_face(vh0,vh1,vh2)

# returns projection of halfedge ('he') vertex; the start vertex if 'useStart',
# otherwise the end vertex.
# 'rotation' dictates projection direction, and refers to angle from XY plane
# 'isFillet' produces simple projection from vertex of 'filletLen' length;
#            if false, performs projection of vertex to XY plane
def halfedgeVertProjection(he,rotation,isFillet=True,filletLen=2,useStart=True):
    # halfedge vertex handles, for accessing mesh points
    startH = oMesh.from_vertex_handle(he)
    endH = oMesh.to_vertex_handle(he)
    
    # actual vertex positions
    startPt = oMesh.point(startH)
    endPt = oMesh.point(endH)
    startGlm = glm.vec3(startPt[0],startPt[1],startPt[2])
    endGlm = glm.vec3(endPt[0],endPt[1],endPt[2])
    
    # vertex positions projected up 5mm toward XY plane
    startZ = startPt[2]+filletLen if isFillet else 0
    endZ = endPt[2]+filletLen if isFillet else 0
    newPt = [startPt[0],startPt[1],startZ] if useStart else [endPt[0],endPt[1],endZ]
    
    # rotate Z-up vector 'rotation' radians around current edge as axis, which will be
    # used to create transitional surface to skirt
    axis = glm.normalize(endGlm - startGlm)
    projDir = glm.normalize(glm.rotate(glm.vec3(0,0,1), rotation, axis))
        
    newPtGlm = glm.vec3(newPt[0],newPt[1],newPt[2])
    startGlm = startGlm if useStart else endGlm        
    projectionDistance = glm.distance(newPtGlm, startGlm)
    if isFillet:
        # projection distance via law of sines for fillets
        projectionDistance = projectionDistance * glm.sin(np.pi/2) / glm.sin(70*np.pi/180)
    else:
        # projection distance via ray-plane intersection for actual skirt
        d = glm.dot(glm.vec3(0,0,-1),projDir)
        projectionDistance = glm.dot((newPtGlm - startGlm),glm.vec3(0,0,-1)) / d
        
    newPtGlm = startGlm + (projectionDistance * projDir)
            
    return [newPtGlm.x,newPtGlm.y,newPtGlm.z]

#
# STEP THREE: create initial fillet with 20 degree wall angle to transition gradually
#             from geometry to skirt
#

# collect boundary edges and the projections of their vertices that form the
# initial fillet (with 20deg wall angle)
rot = 70 * np.pi/180
vertexProjections = dict()
boundaryHEs = []
for he in oMesh.halfedges():
    if oMesh.is_boundary(he):
        boundaryHEs.append(he)

        newPt = halfedgeVertProjection(he,rotation=rot)
 
        vertexProjections[oMesh.from_vertex_handle(he).idx()] = oMesh.add_vertex(newPt)

# "corner faces", eg faces with two boundary edges need to be handled differently
# to avoid possible surface folds when faces project up from highly non-colinear
# edges, which would compromise skirt geometry
cornerFaceHEs = []
for face in oMesh.faces():
    if oMesh.is_boundary(face):
        boundaries = []
        for he in oMesh.fh(face):
            outerHE = oMesh.opposite_halfedge_handle(he)
            if outerHE in boundaryHEs:
                boundaries.append(outerHE)
        if len(boundaries) == 2:
            cornerFaceHEs.append(boundaries)

# return halfedge adjacent to 'he' if it exists on a corner face
def sharedCornerHEs(he):
    for che in cornerFaceHEs:
        if he.idx() == che[0].idx() or he.idx() == che[1].idx():
            return che
    return []

# containers for keeping track of edges that have been added or collapsed and faces
# that need to be added to mesh
processed = []
toAdd = []     
collapsed = []
for he in boundaryHEs:

    if he.idx() in collapsed or he.idx() in processed:
        continue
     
    startH = oMesh.from_vertex_handle(he)
    endH = oMesh.to_vertex_handle(he) 
    sp = oMesh.point(startH)
    ep = oMesh.point(endH)
 
    startPtProjH = vertexProjections[startH.idx()]
    endPtProjH = vertexProjections[endH.idx()]
    
    shared = sharedCornerHEs(he)
    if he in shared:
        # add faces for corner halfedges as well as a triangle that spans the gap
        # between them
        start0H = oMesh.from_vertex_handle(shared[0])
        end0H = oMesh.to_vertex_handle(shared[0])
        start1H = oMesh.from_vertex_handle(shared[1])
        end1H = oMesh.to_vertex_handle(shared[1])
        
        start0Proj = halfedgeVertProjection(shared[0],rotation=rot)
        start0ProjH = oMesh.add_vertex(start0Proj)
        end0ProjH = vertexProjections[end0H.idx()]
        
        start1ProjH = vertexProjections[start1H.idx()]
        end1Proj = halfedgeVertProjection(shared[1],rotation=rot,useStart=False)
        end1ProjH = oMesh.add_vertex(end1Proj)
        
        toAdd.append([start0H,end0H,end0ProjH,start0ProjH])
        toAdd.append([start1H,end1H,end1ProjH,start1ProjH])
        toAdd.append([start0ProjH,end1ProjH,end1H])
        processed.append(shared[0].idx())
        processed.append(shared[1].idx())
        continue

    # collapse edges under given threshold
    # TODO: expose threshold to user via commandline arg
    edgeLength = glm.distance(glm.vec3(sp[0],sp[1],sp[2]),glm.vec3(ep[0],ep[1],ep[2]))
    if edgeLength < 1.5:
        nextHE = oMesh.next_halfedge_handle(he)
        nextShared = sharedCornerHEs(nextHE)
        if nextHE.idx() not in collapsed and nextHE.idx() not in processed and len(nextShared) == 0:
            nextStartH = oMesh.from_vertex_handle(nextHE)
            nextEndH = oMesh.to_vertex_handle(nextHE)
            nextStartPtProjH = vertexProjections[nextStartH.idx()]
            nextEndPtProjH = vertexProjections[nextEndH.idx()]
            toAdd.append([startH,endH,startPtProjH])
            toAdd.append([endH,nextEndH,nextEndPtProjH])
            toAdd.append([startPtProjH,endH,nextEndPtProjH])
            collapsed.append(he.idx())
            collapsed.append(nextHE.idx())
            continue
        
    toAdd.append([startH,endH,endPtProjH,startPtProjH])
    processed.append(he.idx())
    
for ta in toAdd:
    oMesh.add_face(ta)

oMesh.garbage_collection()

#
# STEP FOUR: another fillet! creates another transitional band of faces around the
#            geometry with 45 degree wall angles
#

processed = []
toAdd = []     
collapsed = []
finalVertexProjections = dict()
rot = 45 * np.pi/180
boundaryHEs = []
for he in oMesh.halfedges():
    if oMesh.is_boundary(he):
        boundaryHEs.append(he)

for he in boundaryHEs:
    if he.idx() in collapsed or he.idx() in processed:
        continue
    
    start = oMesh.from_vertex_handle(he)
    end = oMesh.to_vertex_handle(he)
    sp = oMesh.point(start)
    ep = oMesh.point(end)
    
    startProj = halfedgeVertProjection(he,rotation=rot,isFillet=True,filletLen=1)
    endProj = halfedgeVertProjection(he,rotation=rot,isFillet=True,filletLen=1,useStart=False)
    
    startProjH = finalVertexProjections[start.idx()] if start.idx() in finalVertexProjections else oMesh.add_vertex(startProj)
    finalVertexProjections[start.idx()] = startProjH
    endProjH = finalVertexProjections[end.idx()] if end.idx() in finalVertexProjections else oMesh.add_vertex(endProj)
    finalVertexProjections[end.idx()] = endProjH

    # collapse edges under given threshold
    # TODO: expose threshold to user via commandline arg
    edgeLength = glm.distance(glm.vec3(sp[0],sp[1],sp[2]),glm.vec3(ep[0],ep[1],ep[2]))
    if edgeLength < 1.5:
        nextHE = oMesh.next_halfedge_handle(he)
        if nextHE.idx() not in collapsed and nextHE.idx() not in processed:
            nextStartH = oMesh.from_vertex_handle(nextHE)
            nextEndH = oMesh.to_vertex_handle(nextHE)
            nextStartProj = halfedgeVertProjection(nextHE,rotation=rot,isFillet=True,filletLen=1)
            nextEndProj = halfedgeVertProjection(nextHE,rotation=rot,isFillet=True,filletLen=1,useStart=False)
            nextStartProjH = finalVertexProjections[nextStartH.idx()] if nextStartH.idx() in finalVertexProjections else oMesh.add_vertex(nextStartProj)
            nextEndProjH = finalVertexProjections[nextEndH.idx()] if nextEndH.idx() in finalVertexProjections else oMesh.add_vertex(nextEndProj)
            # add to final verts dict so subsequent faces refer to any existing projections
            finalVertexProjections[nextStartH.idx()] = nextStartProjH
            finalVertexProjections[nextEndH.idx()] = nextEndProjH
            
            toAdd.append([start,end,startProjH])
            toAdd.append([end,nextEndH,nextEndProjH])
            toAdd.append([startProjH,end,nextEndProjH])
            collapsed.append(he.idx())
            collapsed.append(nextHE.idx())
            continue

    toAdd.append([start,end,endProjH,startProjH])
    processed.append(he.idx())

for ta in toAdd:
    oMesh.add_face(ta)
oMesh.garbage_collection()

#
# STEP FIVE: create skirt - final band of faces projected up to XY plane from geometry,
#            to create a final 70 degree wall angle
#

rot = 20 * np.pi/180
vertexProjections = dict()
for he in oMesh.halfedges():
    if oMesh.is_boundary(he):
        start = oMesh.from_vertex_handle(he)
        end = oMesh.to_vertex_handle(he)
        
        startProj = halfedgeVertProjection(he,rotation=rot,isFillet=False)
        endProj = halfedgeVertProjection(he,rotation=rot,isFillet=False,useStart=False)
        
        startProjH = finalVertexProjections[start.idx()] if start.idx() in finalVertexProjections else oMesh.add_vertex(startProj)
        finalVertexProjections[start.idx()] = startProjH
        endProjH = finalVertexProjections[end.idx()] if end.idx() in finalVertexProjections else oMesh.add_vertex(endProj)
        finalVertexProjections[end.idx()] = endProjH
        oMesh.add_face([start,end,endProjH,startProjH])
oMesh.garbage_collection()

#
# FINAL STEP: triangulate reoriented, filleted, and skirted mesh, then write it to file
#

oMesh.triangulate()
om.write_mesh("input_skirted.stl", oMesh)
