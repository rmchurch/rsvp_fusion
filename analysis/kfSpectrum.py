# function[kfspec,k,f]=kfSpectrum(L,time,frames,varargin)
# %kfspec - Calculates the spatial wavenumber and temporal frequency
# %spectrum, in the continuous form as:
# %   ??dl dt f(l,t)*exp(-j*(2*pi*f*t - k*l))
# %(notice opposite sign in front of k. This means fft2 won't work directly.
# %Instead, we use ifft and fft).
# %
# %INPUTS
# %   L           Distance array [m]
# %   time        Time array [s]
# %   frames      Data to transform, size is [NL,Ntime,...]
# %
# %OUTPUTS
# %   kfspec      Fourier transformed k,f spectrum, S(k|f)
# %   k           Wavenumbers in [m^-1]
# %   f           Frequency in [Hz]
# %KEYWORDS
# %   noNormalize Output S(k,f) instead of S(k|f)=S(k,f)/S(f)
# %   noFilter    Don't do an average filter on the data (used for smoothing)
# %   window      Apply a hanning window

import numpy as np
from scipy.ndimage.filters import gaussian_filter

def kfSpectrum(L,time,frames,noNormalize=False, noFilter=False, window=None):

	if window is not None:
		#default to Hanning, add others later
	    wL=0.5*(1-np.cos(2*np.pi*np.arange(L.size)))
	    wTime=0.5*(1-np.cos(2*np.pi*(np.arange(time.size))))
	    win2 = wL[:,np.newaxis]*wTime[np.newaxis,:] #or should this be matrix mult?
	    frames = frames*win2

	NFFT = 2**np.ceil(np.log2(frames.shape[0:2])).astype(int)
	# %%%Create k and f arrays
	# %create frequency array
	Fs=1./np.mean(np.diff(time));
	f=Fs/2*np.linspace(0,1,NFFT[1]/2+1) # %highest frequency 0.5 sampling rate (Nyquist)

	# %METHOD 1: No anti-aliasing
	# %create k array
	kmax=np.pi/np.min(np.diff(L))
	k=kmax*np.linspace(-1,1,NFFT[0])
	kfspec=np.fft.fftshift(np.fft.ifft(np.fft.fft(frames,n=NFFT[1],axis=1),n=NFFT[0],axis=0))  #%fftshift since Matlab puts positive frequencies first
	kfspec=kfspec[:,NFFT[1]/2-1:,...]

	# % %METHOD 2: Anti-aliasing
	# % kfspec=fftshift(fft(frames,NFFT(2),2));
	# % %for now, remove negative frequency components
	# % kfspec=kfspec(:,NFFT(2)/2:end);
	# % kmin=pi/(L(end)-L(1));
	# % kmax=pi/min(diff(L))*0.85;
	# % k=[linspace(-kmax,-kmin,NFFT(1)/2-1) 0 linspace(kmin,kmax,NFFT(1)/2)];
	# % kfspec=exp(i*k(:)*L(:)')*kfspec;

	# %%%normalize, S(k|w)=S(k,w)/S(w)
	if not noNormalize:
	    kfspec=np.abs(kfspec)/np.sum(np.abs(kfspec),axis=0)[np.newaxis,:]

	# %%%OPTIONAL: filtering (smooths images)
	if not noFilter:
		kfspec = gaussian_filter(np.abs(kfspec), sigma=5)

	return k,f,kfspec