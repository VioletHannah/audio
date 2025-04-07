import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

os.putenv('CUDA_VISIBLE_DEVICES', '')


def mel2freq(input_mel_freq):
    freq = (10 ** (input_mel_freq / 2595.) - 1.) * 700

    return freq


def freq2mel(input_freq):
    mel_freq = 2595. * np.log10(1 + input_freq / 700.)

    return mel_freq


def mel_filters_numpy():
    mel_filters = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=256,
        num_spectrogram_bins=2048 // 2 + 1,
        sample_rate=24000,
        lower_edge_hertz=10,
        upper_edge_hertz=12000 - 10)

    mel_filters = tf.transpose(mel_filters, [1, 0])

    mel_filters = mel_filters.numpy()
    sqrt_filters = np.sqrt(mel_filters)
    center_freqs = np.argmax(sqrt_filters, axis=1)
    center_freqs = center_freqs.astype(np.float32)
    peaks = np.amax(sqrt_filters, axis=1, keepdims=True)
    half_magnitude = peaks / 2.
    fwhms = sqrt_filters > half_magnitude
    fwhms = fwhms.astype(np.float32)
    fwhms = np.sum(fwhms, axis=1)

    return center_freqs, fwhms


def mel_filters_tf():
    mel_filters = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=256,
        num_spectrogram_bins=2048 // 2 + 1,
        sample_rate=24000,
        lower_edge_hertz=0,
        upper_edge_hertz=12000)
    n_fft = 2048
    coeff = tf.math.sqrt(2. * tf.math.log(2.)) * n_fft

    mel_filters = tf.transpose(mel_filters, [1, 0])

    sqrt_filters = tf.math.sqrt(mel_filters)
    center_frequencies = tf.cast(
        tf.argmax(sqrt_filters, axis=1), dtype=tf.float32)
    peaks = tf.reduce_max(sqrt_filters, axis=1, keepdims=True)
    half_magnitudes = peaks / 2.
    fwhms = tf.reduce_sum(
        tf.cast(sqrt_filters >= half_magnitudes, dtype=tf.float32), axis=1)

    fwhms = coeff / (np.pi * fwhms)
    fwhms = fwhms/2.
    center_frequencies = center_frequencies * 2 * np.pi / n_fft

    return center_frequencies, fwhms

def log_curve_fitting(center_freqs, period_num, log10=True):
    '''
    Fitting a logarithmic curve
    :param X: the x data
    :param Y: the y data
    :param log10: boolean, if to fit with log10, otherwise with np.log()
    :return: the fitting parameter, a, b
    '''
    plt.scatter(center_freqs, period_num)
    plt.xlabel('Frequency Response', fontsize=14)
    plt.ylabel('Periodicity', fontsize=14)

    n = len(center_freqs)
    x_bias = np.ones((n, 1))

    X = np.reshape(center_freqs, (n, 1))
    Y = period_num

    if log10:
        X_log = np.log10(X)
    else:
        X_log = np.log(X)

    x_new = np.append(x_bias, X_log, axis=1)
    x_new_transpose = np.transpose(x_new)

    x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)

    temp1 = np.linalg.inv(x_new_transpose_dot_x_new)
    temp2 = x_new_transpose.dot(Y)

    theta = temp1.dot(temp2)

    shift = theta[0]
    slope = theta[1]

    if log10:
        Y_plot = shift + slope * np.log10(X)
    else:
        Y_plot = shift + slope * np.log(X)

    plt.scatter(X, Y)
    plt.plot(X, Y_plot, c="b", linewidth=2)

    from sklearn.metrics import r2_score
    Accuracy = r2_score(Y, Y_plot)
    print('Fitting Accuracy: {}'.format(Accuracy))
    print('shift = {}, slope = {}'.format(shift, slope))

    plt.savefig('curve_fit.svg')
    plt.show()
    # plt.savefig('curve_fit.svg')
    exit(0)

    import pdb
    pdb.set_trace()
    return slope, shift


def gabor_impulse_response(t, center, fwhm):
    """Computes the gabor impulse response."""
    denominator = 1.0 / (tf.math.sqrt(2.0 * math.pi) * fwhm)
    gaussian = tf.exp(tf.tensordot(1.0 / (2. * fwhm ** 2), -t ** 2, axes=0))
    center_frequency_complex = tf.cast(center, tf.complex64)
    t_complex = tf.cast(t, tf.complex64)
    sinusoid = tf.math.exp(
        1j * tf.tensordot(center_frequency_complex, t_complex, axes=0))
    denominator = tf.cast(denominator, dtype=tf.complex64)[:, tf.newaxis]
    gaussian = tf.cast(gaussian, dtype=tf.complex64)

    gauss = gaussian.numpy()
    # plt.plot(gauss[0,:])
    # plt.show()
    center_freqs = center.numpy()
    periodicity = 1. / center_freqs
    period_num = np.divide(fwhm.numpy(), periodicity)

    print('period_num = \n{}'.format(period_num))

    # import pdb
    # pdb.set_trace()
    center_freqs_to_fit = center_freqs * 2048 / (2 * np.pi)

    a, b = log_curve_fitting( center_freqs=center_freqs_to_fit, period_num=period_num, log10=True )
    # plt.scatter( center_freqs, period_num, alpha=0.6 )
    # plt.show()
    # import pdb
    # pdb.set_trace()

    # return sinusoid

    return denominator * sinusoid * gaussian


def get_gabor_filter_bank():
    center_freqs, fwhms = mel_filters_tf()
    size = 401
    t = tf.range(-(size // 2), (size + 1) // 2, dtype=tf.float32)

    # print(center_freqs)
    gabor_filter_bank = gabor_impulse_response(t, center_freqs, fwhms)

    return gabor_filter_bank


def double_check():
    num_mel_bins = 40
    num_spectrogram_bins = 512 // 2 + 1
    sample_rate = 16000
    lower_edge_hertz = 0
    upper_edge_hertz = 8000

    mel_low = freq2mel(lower_edge_hertz)
    mel_high = freq2mel(upper_edge_hertz)
    m_range = np.arange(num_mel_bins).astype(np.float32)

    mel_scale_freq_vec = mel_low + m_range * (
            (mel_high - mel_low) / (num_mel_bins + 1))

    mel_scale_freq = mel2freq(mel_scale_freq_vec)

    mel_scale_freq = (512 / sample_rate) * mel_scale_freq

    center_freq_tf = mel_filters_tf()


def main():
    gabor_filter_bank = get_gabor_filter_bank()
    gabor_filter_bank = gabor_filter_bank.numpy()
    plt.plot(gabor_filter_bank[250, :])
    plt.show()


if __name__ == '__main__':
    main()
