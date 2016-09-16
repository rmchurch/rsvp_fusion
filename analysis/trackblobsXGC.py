#/usr/bin python

#blobParams
#   X0_in x
#   Y0_in x
#   Parent
#   Child
#   velr
#   velTheta
#   area x
#   thetaHat

import numpy as np

def trackblobsXGC(time,blobParams,isTwoSpeed=False):

    if isTwoSpeed:
        maxJumpPol = np.mean(np.diff(time))*40e3 #don't allow blobs moving poloidally faster than 40km/s
        maxJumpRad=np.mean(np.diff(time))*5e3 #don't allow blobs moving radially faster than 5km/s
    else:
        maxJump = np.mean(np.diff(time))*20e3 #don't allow blobs moving poloidally faster than 20km/s

    maxAreaChange = 100

    #TODO: For now, assume blobParams originally created with Parent,Child,velr,velTheta
    #add the Parent and Child field
    # for t=1:length(time)
    #     if ~isempty(blobParams{t})
    #         [blobParams{t}.Parent]=deal(NaN);
    #         [blobParams{t}.Child]=deal(NaN);
    #         [blobParams{t}.velr]=deal(NaN);
    #         [blobParams{t}.velTheta]=deal(NaN);

    for t in range(time.size-1):
        if (blobParams[t] is None) | (blobParams[t+1] is None): continue

        Nthis=blobParams[t].size
        Nnext=blobParams[t+1].size
	if (Nthis==0) | (Nnext==0): continue
        X0this=np.tile(blobParams[t]['R0'][:,np.newaxis],(1,Nnext))
        Y0this=np.tile(blobParams[t]['Z0'][:,np.newaxis],(1,Nnext))
        X0next=np.tile(blobParams[t+1]['R0'][np.newaxis,:],(Nthis,1))
        Y0next=np.tile(blobParams[t+1]['Z0'][np.newaxis,:],(Nthis,1))
        
        dR=X0next-X0this
        dZ=Y0next-Y0this
        
        distMat=np.sqrt(dR**2.+dZ**2.)
        
        #calculate the change in area
        areaThis=np.tile( blobParams[t]['area'][:,np.newaxis],(1,Nnext))
        areaNext=np.tile( blobParams[t+1]['area'][np.newaxis,:],(Nthis,1))
        areaChange = np.abs(areaNext - areaThis)

        if isTwoSpeed:
            #%get the radial and poloidal direction vectors
            thetaHat=blobParams[t]['thetaHat']
            rHat=np.dot(np.array([[0,1],[-1,0]]),thetaHat.T).T
            #%calculate radial and poloidal distance travelled (really distance
            #%tangent to flux surface)
            dr=dR*np.tile(rHat[:,0][:,np.newaxis],(1,Nnext)) + dZ*np.tile(rHat[:,1][:,np.newaxis],(1,Nnext))
            dTheta=dR*np.tile(thetaHat[:,0][:,np.newaxis],(1,Nnext)) + dZ*np.tile(thetaHat[:,1][:,np.newaxis],(1,Nnext))
            VrThis=np.tile(blobParams[t]['velr'][:,np.newaxis],(1,Nnext))
            VthetaThis=np.tile(blobParams[t]['velTheta'][:,np.newaxis],(1,Nnext))
            VrNext=dr/(time[t+1]-time[t])
            VthetaNext=dTheta/(time[t+1]-time[t])
            distr=np.abs(dr)
            distTheta=np.abs(dTheta)
            
            #%determine the blob combinations that are too far
            badInds = np.where( (distr>maxJumpRad) | (distTheta>maxJumpPol) | \
                           (np.isnan(distr)) | (np.isnan(distTheta)) | \
                           (np.sign(VthetaThis)!=np.sign(VthetaNext)) & (np.abs(VthetaThis-VthetaNext)>maxJumpPol/2) | \
                           (areaChange > maxAreaChange) )
        else:
            badInds = np.where( (distMat>maxJump) | (areaChange>maxAreaChange) )

        #remove the combinations that don't meet the criteria from
        distMat[badInds] = np.nan

        #%now determine for each blob in future frame which current frame blob
        #%they are closest to. The rest of the current frame blobs are
        #%childless.
        if Nnext<=Nthis: #%if next frame has fewer blobs
            ax = 0
        else:
            ax = 1
            
        minVals=np.min(distMat,axis=ax)
        minInds = distMat.argmin(axis=ax)
        #remove NaN
        goodInds=np.where( ~np.isnan(minVals))[0]
        minInds=minInds[goodInds]
        if minInds.size>0:
            if ax:
                inds1 = goodInds
                inds2 = minInds
            else:
                inds1 = minInds
                inds2 = goodInds

            #%assign parent and child
            blobParams[t+1][inds2]['Parent']=inds1
            blobParams[t][inds1]['Child']=inds2
            
            #%calculate blob quantities
            blobParams[t+1][inds2]['velr']=dr[inds1,inds2]/(time[t+1]-time[t])
            blobParams[t+1][inds2]['velTheta']=dTheta[inds1,inds2]/(time[t+1]-time[t])
    return blobParams
