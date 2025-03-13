import tensorflow as tf

def load_and_mel_file(path_tensor):
    #From: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
    pcm, sample_rate = tf.audio.decode_wav(path_tensor)
    sr_f = tf.cast(sample_rate, tf.float32) #Mismatch in types between output of decode_wav and input to linear_to_mel_weight_matrix
    print(pcm, sample_rate, sr_f)

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256,
                           fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr_f, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    print(log_mel_spectrograms)
    
    return log_mel_spectrograms

path_ds = tf.data.Dataset.list_files("*.wav")
mel_ds = path_ds.map(load_and_mel_file)

