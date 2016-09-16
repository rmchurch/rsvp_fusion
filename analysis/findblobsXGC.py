#/usr/bin python

#findblobsXGC
import matplotlib.pyplot as plt
import numpy as np

def contour_data(R,Z,C,data,remove_empty=False):
  #first, determine how many contour paths were generated
  Ncont = 0
  for collection in C.collections:
    Ncont += len(collection.get_paths())

  n = 0
  R0 = np.empty((Ncont,))
  Z0 = np.empty((Ncont,))
  area = np.empty((Ncont,))
  levs = np.empty((Ncont,))
  maxVals = np.empty((Ncont,))
  minVals = np.empty((Ncont,))
  medVals = np.empty((Ncont,))
  inds = np.empty((Ncont,),dtype=object)
  paths = np.empty((Ncont,),dtype=object)
  for (k,collection) in enumerate(C.collections):
    pathsk = collection.get_paths()
    for path in pathsk:
      RZcont = path.to_polygons()[0]
      if np.all(RZcont[0,:] != RZcont[-1,:]): pass #not a closed contour

      minR = RZcont[:,0].min(); maxR = RZcont[:,0].max()
      minZ = RZcont[:,1].min(); maxZ = RZcont[:,1].max()
      rzinds = (R>=minR) & (R<=maxR) & (Z>=minZ) & (Z<=maxZ)
      subdata = data[rzinds]

      #center positions
      R0[n] = np.mean(RZcont[:,0])
      Z0[n] = np.mean(RZcont[:,1])
      #area (Green's thm for area, A = \int_l x*dy)
      area[n] = np.sum(RZcont[:-1,0]*np.diff(RZcont[:,1])) 
      if (subdata.size==0) & (remove_empty): pass
      if subdata.size == 0:
        maxVals[n] = levs[n]
        minVals[n] = levs[n]
        medVals[n] = levs[n]
      else:
        inside = path.contains_points( np.vstack((R[rzinds],Z[rzinds])).T )
        if not np.any(inside):
          maxVals[n] = levs[n]
          minVals[n] = levs[n]
          medVals[n] = levs[n]
        else:
          maxVals[n] = np.max(np.abs(subdata[inside]))
          minVals[n] = np.min(np.abs(subdata[inside]))
          medVals[n] = np.median(np.abs(subdata[inside]))
          inds[n] = np.where(rzinds)[0][inside]
      levs[n] = C.cvalues[k]
      paths[n] = path
      n += 1
  return (R0[0:n],Z0[0:n],area[0:n],levs[0:n],minVals[0:n],maxVals[0:n],medVals[0:n],inds[0:n],paths[0:n])


def findblobsXGC(R,Z,tri,data,blobHt=1.05,holeHt=0.95, \
							  blob_levels=None,hole_levels=None, \
							  min_rise = None,remove_empty=True):

	if blob_levels is None: blob_levels = np.array([blobHt])
	if hole_levels is None: hole_levels = np.array([holeHt])
	if min_rise is None: min_rise = 0.5*  (blobHt - 1)
	Cblobs = plt.tricontour(R,Z,tri,data,levels=blob_levels)
	Choles = plt.tricontour(R,Z,tri,data,levels=hole_levels)

	R0blobs,Z0blobs,areaBlobs,blobLevs,blobMinVals,blobMaxVals,blobMedVals,blobInds,blobPaths = contour_data(R,Z,Cblobs,data,remove_empty=remove_empty)
	R0holes,Z0holes,areaHoles,holeLevs,holeMinVals,holeMaxVals,holeMedVals,holeInds,holePaths = contour_data(R,Z,Choles,data,remove_empty=remove_empty)

	isBlobCandidate = (blobMedVals >= blobHt) & ((blobMaxVals - blobLevs) > min_rise)
	isHoleCandidate = (holeMedVals >= holeHt) & ((holeMaxVals - holeLevs) > min_rise)

	R0blobs = R0blobs[isBlobCandidate]
	Z0blobs = Z0blobs[isBlobCandidate]
	areaBlobs = areaBlobs[isBlobCandidate]
	blobInds = blobInds[isBlobCandidate]
	blobPaths = blobPaths[isBlobCandidate]
	
	R0holes = R0holes[isHoleCandidate]
	Z0holes = Z0holes[isHoleCandidate]
	areaHoles = areaHoles[isHoleCandidate]
	holeInds = holeInds[isHoleCandidate]
	holePaths = holePaths[isHoleCandidate]

	blobParams = np.empty(R0blobs.size,dtype={'names':['R0', 'Z0','area','thetaHat','Parent', 'Child','velr','velTheta'], \
                               'formats':['f4','f4','f4',('f4',(2,)), 'i4', 'i4','f4','f4']})
	#if R0blobs.size>1: blobParams[:] = np.nan
	blobParams['R0'] = R0blobs
	blobParams['Z0'] = Z0blobs
	blobParams['area'] = areaBlobs
	holeParams = np.empty(R0holes.size,dtype={'names':['R0', 'Z0','area','thetaHat','Parent', 'Child','velr','velTheta'], \
                               'formats':['f4','f4','f4',('f4',(2,)), 'i4', 'i4','f4','f4']})
	#if R0holes.size>1: holeParams[:] = np.nan
	holeParams['R0'] = R0holes
	holeParams['Z0'] = Z0holes
	holeParams['area'] = areaHoles
	#TODO: Add ellipse parameter determination
	return (blobInds,blobPaths,blobParams,holeInds,holePaths,holeParams)
