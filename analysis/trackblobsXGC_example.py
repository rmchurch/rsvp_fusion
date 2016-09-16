#/usr/bin python

from findblobsXGC import findblobsXGC
from trackblobsXGC import trackblobsXGC
import adios as ad
import numpy as np
from matplotlib.tri import Triangulation,LinearTriInterpolator
from IPython.parallel import Client

rc = Client()
dview = rc[:]
with dview.sync_imports(): #these required by findblobsXGC
        import matplotlib.pyplot as plt
        import numpy as np
        from findblobsXGC import findblobsXGC

#get data from f3d
fileDir = '/ccs/home/rchurchi/scratch/ti252_ITER_new_profile/'
#mesh
fm = ad.file(fileDir + 'xgc.mesh.bp')
RZ = fm['/coordinates/values'][...]
tri = fm['nd_connect_list'][...]
psi = fm['psi'][...]
psi_x=    11.10093394162000
psin = psi/psi_x
eq_x_z=   -3.442893939000000
fm.close()
spaceinds = (psin>0.95) & (psin<1.05) & ( (RZ[:,1]>=eq_x_z) | (psin>=1) ) 

tmp=spaceinds[tri] #rzspaceinds T/F array, same size as R
goodTri=np.all(tmp,axis=1) #only use triangles who have all vertices in rzInds
tri=tri[goodTri,:]
#remap indices in triangulation
indices=np.where(spaceinds)[0]
#for i in range(len(indices)):
#        tri[tri==indices[i]]=i
imap = np.empty((indices.max()+1))
imap[indices] = np.arange(0,indices.size)
triGrid = imap[tri]
Rgrid = RZ[spaceinds,0]
Zgrid = RZ[spaceinds,1]
psinGrid = psin[spaceinds]
print 'Mesh loaded'

#bfield
fb = ad.file(fileDir + 'xgc.bfield.bp')
bfield = fb['/node_data[0]/values'][spaceinds,:]
fb.close()

#tindex
f1d = ad.file(fileDir+'xgc.oneddiag.bp')
time = np.unique(f1d['time'][:])[50:]
tindex = np.unique(f1d['tindex'][:])[50:] #remove first 50
f1d.close()
Ntimes = tindex.size
Nplanes = 32

triObj = Triangulation(Rgrid,Zgrid,triGrid)
fBR = LinearTriInterpolator(triObj,bfield[:,0])
fBZ = LinearTriInterpolator(triObj,bfield[:,1])

#put the required things into the parallel workers
#dview.push(dict(Rgrid=Rgrid,Zgrid=Zgrid,triGrid=triGrid))

## find blobs in each plane, time
#blobInds = np.empty((Nplanes,Ntimes),dtype=object)
#blobPaths = np.empty((Nplanes,Ntimes),dtype=object)
#blobParams = np.empty((Nplanes,Ntimes),dtype=object)
#holeInds = np.empty((Nplanes,Ntimes),dtype=object)
#holePaths = np.empty((Nplanes,Ntimes),dtype=object)
#holeParams = np.empty((Nplanes,Ntimes),dtype=object)
#for (it,t) in enumerate(tindex):
#    print 'Starting time ind '+str(t)
#    try:
#        f3d = ad.file(fileDir+'xgc.f3d.'+str(t).zfill(5)+'.bp')
#    except Exception as e:
#	print e
#	continue
#    ne = f3d['e_den'][spaceinds,:]
#    f3d.close()
#
#    ne0 = np.mean(ne,axis=1)
#    data = ne/ne0[:,np.newaxis]
#    #out = dview.map_sync(lambda d: findblobsXGC(Rgrid,Zgrid,triGrid,d,blobHt=1.02,holeHt=0.98),np.rollaxis(data,-1))
#    out = dview.map_sync(lambda d: findblobsXGC(Rgrid,Zgrid,triGrid,d),np.rollaxis(data,-1))
#    out = np.array(out)
#    blobInds[:,it] = out[:,0]
#    blobPaths[:,it] = out[:,1]
#    blobParams[:,it] = out[:,2]
#    holeInds[:,it] = out[:,3]
#    holePaths[:,it] = out[:,4]
#    holeParams[:,it] = out[:,5]
#    # for p in range(Nplanes):
#    #   data = neOverne0[:,p,t]
#    #   blobInds[p,t],blobPaths[p,t],blobParams[p,t],\
#    #   holeInds[p,t],holePaths[p,t],holeParams[p,t] = findblobsXGC(Rgrid,Zgrid,triGrid,data)
#
#np.savez('trackBlobs_example_preTracking.npz',Rgrid=Rgrid,Zgrid=Zgrid,triGrid=triGrid,psinGrid=psinGrid,spaceinds=spaceinds,\
#                                  tindex=tindex,\
#                                  blobInds=blobInds,blobPaths=blobPaths,blobParams=blobParams,\
#                                  holeInds=holeInds,holePaths=holePaths,holeParams=holeParams)

f = np.load('trackBlobs_example_preTracking.npz')
blobParams = f['blobParams']
holeParams = f['holeParams']



## populate the thetaHat unit vector
for t in range(Ntimes):
    for p in range(Nplanes):
        if blobParams[p,t] is not None:
		BR = fBR(blobParams[p,t]['R0'],blobParams[p,t]['Z0'])
        	BZ = fBZ(blobParams[p,t]['R0'],blobParams[p,t]['Z0'])
       		Bpol = np.vstack((BR,BZ)).T
        	blobParams[p,t]['thetaHat'] = Bpol / np.sqrt(np.sum(Bpol**2.,axis=1)[:,np.newaxis])

dview.push(dict(time=time,blobParams=blobParams,holeParams=holeParams))
with dview.sync_imports(): #these required by findblobsXGC
        from trackblobsXGC import trackblobsXGC

out = dview.map_sync(lambda d: trackblobsXGC(time,d,isTwoSpeed=True),blobParams)
blobParams = np.array(blobParams)
out = dview.map_sync(lambda d: trackblobsXGC(time,d,isTwoSpeed=True),holeParams)
holeParams = np.array(holeParams)

#for p in range(Nplanes):
#    blobParams[p,:] = trackblobsXGC(time,blobParams[p,:],isTwoSpeed=True)
#    holeParams[p,:] = trackblobsXGC(time,holeParams[p,:],isTwoSpeed=True)

np.savez('trackBlobs_example_finalParams.npz',blobParams=blobParams,holeParams=holeParams)

