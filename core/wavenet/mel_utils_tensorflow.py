import numpy as np
import tensorflow as tf
import decimal
import math

logSpecDbConst = 10.0 / math.log(10.0) * math.sqrt(2.0)
def mcd(mfcc1, mfcc2):
    '''
    computes the mel cepstral distance between 2 signals given the mel frequency cepstral coefficients (mfcc) of the signals
    Args:
        mfcc1: mfcc for first signal
        mfcc2: mfcc for the second signal

    Returns:

    '''
    diff = mfcc1 - mfcc2
    diff = tf.squeeze(diff)
    return logSpecDbConst * tf.reduce_mean(tf.sqrt(tf.abs(tf.tensordot(diff, tf.transpose(diff), axes=1))))


def logSpecDb_mcd(mfcc1, mfcc2):
    return logSpecDbConst * mcd(mfcc1, mfcc2)


def tf_log10(x):
    '''
    computes the log in base 10 of the tensor x
    Args:
        x (tensorflow.Tensor): the tensor on whic to compute the log in base 10

    Returns:
        tensorflow.Tensor: a tensorflow node of the log in base 10
    '''
    return tf.log(x + 1e-6) / tf.log(10)


def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.

    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def mfcc_tf(pcm, sample_rate, win_len=0.025, winstep=0.01):
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    # pcm = tf.compat.v1.placeholder(tf.float32, [None, None])
    pcm = tf.squeeze(pcm, axis=-1)
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    frame_length = int(win_len * sample_rate)
    frame_step = int(sample_rate * winstep)
    stfts = tf.signal.stft(pcm, frame_length=frame_length, frame_step=frame_step, fft_length=400)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0, sample_rate / 2, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs_tf = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :13]
    return mfccs_tf


def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97, appendEnergy=True,
         winfunc=lambda x: np.ones((x,))):
    """Compute MFCC features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use np window functions here e.g. winfunc=np.hamming
    :returns: A np array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    feat = tf.log(feat + 1e-6)
    feat = tf.signal.dct(feat, type=2, axis=-1, norm='ortho')[..., :numcep]
    if appendEnergy:
        # replace first cepstral coefficient with log of frame energy
        energy = tf.expand_dims(energy, axis=-1)
        feat = tf.concat([tf.log(energy + 1e-6), feat[..., 1:]], axis=-1)

    return feat


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
          winfunc=lambda x: np.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use np window functions here e.g. winfunc=np.hamming
    :returns: 2 values. The first is a np array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = tf.reduce_sum(pspec, axis=-1)  # this stores the total energy in each frame
    tf_epsilon = tf.ones_like(energy) * tf.keras.backend.epsilon()
    energy = tf.where(energy == 0, tf_epsilon, energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = tf.tensordot(pspec, tf.transpose(fb), axes=1)  # compute the filterbank energies
    tf_epsilon = tf.ones_like(feat) * tf.keras.backend.epsilon()
    feat = tf.where(feat == 0, tf_epsilon, feat)  # if feat is zero, we get problems with log

    return feat, energy


def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
             nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
             winfunc=lambda x: np.ones((x,))):
    """Compute log Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use np window functions here e.g. winfunc=np.hamming
    :returns: A np array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    return tf.log(feat + 1e-6)


def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = tf.zeros([nfilt, nfft // 2 + 1], tf.float32)
    false_value = -1

    mask = np.ones(fbank.shape) * false_value
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            mask[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            mask_tf = tf.constant(mask, dtype=tf.float32)
            fbank = tf.where(mask_tf >= 0, mask_tf, fbank)
            mask[j, i] = false_value
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            mask[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
            mask_tf = tf.constant(mask, dtype=tf.float32)
            fbank = tf.where(mask_tf >= 0, mask_tf, fbank)
            mask[j, i] = false_value
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = tf.arange(ncoeff)
        lift = 1 + (L / 2.) * tf.sin(tf.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


###########
# SIGPROC #
###########

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    winfunc = lambda x: tf.ones((x,))
    sig_shape = tf.shape(sig)
    batch, slen = sig_shape[0], sig_shape[1]
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))

    val1 = lambda: 1
    val2 = lambda: tf.cast(1 + tf.ceil((1 * slen - frame_len) / frame_step), tf.int32)
    numframes = tf.case([(slen <= frame_len, val1)], default=val2)

    padlen = tf.cast((numframes - 1) * frame_step + frame_len, tf.int32)
    zeros = tf.zeros((batch, padlen - slen,), dtype=sig.dtype)
    padsignal = tf.concat((sig, zeros), axis=1)

    indices = tf.tile(tf.expand_dims(tf.range(0, frame_len), axis=0), (numframes, 1))
    indices = indices + tf.transpose(
        tf.tile(tf.expand_dims(tf.range(0, numframes * frame_step, frame_step), axis=0), (frame_len, 1)))

    frames = tf.gather(padsignal, indices, axis=-1)
    win = tf.tile(tf.expand_dims(winfunc(frame_len), axis=0), (numframes, 1))

    return frames * win


def magspec_np(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = (np.fft.rfft(frames, NFFT))
    return np.absolute(complex_spec).astype('float64')


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    mag_spec = tf.numpy_function(magspec_np, [frames, NFFT], tf.float64)

    mag_spec = tf.cast(mag_spec, tf.float32)
    return 1.0 / NFFT * tf.square(mag_spec)


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    batch = tf.shape(signal)[0]
    # new_signal = signal, dtype=tf.float32)
    first_elem = tf.reshape(signal[:, 0], (batch, -1, 1))
    new_signal = tf.concat((first_elem, signal[:, 1:] - coeff * signal[:, :-1]), axis=-2)

    return tf.reshape(new_signal, (batch, -1))
