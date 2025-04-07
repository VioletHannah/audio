import numpy as np
import math
import os
import torch
# import tensorflow as tf
import torch.nn as nn
import matplotlib.pyplot as plt


class MelScaleFilterBank(nn.Module):
    def __init__(self, n_fft=2048, kernel_length=1025, filter_num=256,
                 sample_rate=24000):
        '''
        我们按照 LEAF 的论文来构建美尔尺度滤波器组。
        :param n_fft: n_fft
        :param kernel_length: kernel_length
        :param filter_num: the number of filters
        :param sample_rate: sampling rate
        '''
        super(MelScaleFilterBank, self).__init__()
        assert kernel_length % 2 == 1
        self.n_fft = n_fft
        self.kernel_length = kernel_length
        self.filter_num = filter_num
        self.sample_rate = sample_rate

    def mel2freq(self, input_mel_freq):
        freq = (10 ** (input_mel_freq / 2595.) - 1.) * 700

        return freq

    def freq2mel(self, input_freq):
        mel_freq = 2595. * np.log10(1 + input_freq / 700.)

        return mel_freq

    def init_learnable_centerfreqs_fwhms(self):
        center_freqs, fwhms = self.get_centerfreqs_fwhms()
        self.center_freqs = nn.Parameter(
            torch.Tensor(center_freqs).to(torch.float32))
        self.fwhms = nn.Parameter(torch.Tensor(fwhms).to(torch.float32))

    def get_centerfreqs_fwhms(self):
        mel_filters = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.filter_num,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=0,
            upper_edge_hertz=self.sample_rate // 2)
        n_fft = self.n_fft
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
        center_freqs = center_frequencies * 2 * np.pi / n_fft

        center_freqs = center_freqs.numpy()
        fwhms = fwhms.numpy()

        return torch.Tensor(center_freqs).to(torch.float32), torch.Tensor(
            fwhms).to(torch.float32)

    def get_impulse_response_np(self, t, center_freqs, fwhms):
        """
        Compute Filter Response
        :param t: the temporal length vector
        :param center_freqs: center freqs
        :param fwhms: fwhms
        :return: constructed filter bank
        """
        # center_freqs = center_freqs
        denominator = 1.0 / (np.sqrt(2.0 * math.pi) * fwhms)
        gaussian = np.exp(
            np.tensordot(1.0 / (2. * fwhms ** 2), -t ** 2, axes=0))
        center_freqs = center_freqs.astype(np.complex64)
        t = t.astype(np.complex64)
        sinusoid = np.exp(1j * np.tensordot(center_freqs, t, axes=0))
        denominator = denominator.astype(np.complex64)[:, np.newaxis]
        gaussian = gaussian.astype(np.complex64)

        return denominator * sinusoid * gaussian

    def get_impuse_response(self, t, center_freqs, fwhms):
        denominator = 1.0 / (np.sqrt(2.0 * math.pi) * fwhms)
        gaussian = torch.exp(
            torch.tensordot((1.0 / (2. * fwhms ** 2)).view((-1, 1)),
                            (-t ** 2).view((1, -1)),
                            dims=([1], [0])))

        center_freqs = center_freqs.to(torch.complex64)
        t = t.to(torch.complex64)
        sinusoid = torch.exp(1j * torch.tensordot(center_freqs.view((-1, 1)),
                                                  t.view((1, -1)),
                                                  dims=([1], [0])))
        denominator = denominator.to(torch.complex64)
        denominator = torch.unsqueeze(denominator, dim=1)
        gaussian = gaussian.to(torch.complex64)

        return denominator * sinusoid * gaussian

    def get_melscale_filterbank(self):
        center_freqs, fwhms = self.get_centerfreqs_fwhms()
        t = torch.arange(start=-(self.kernel_length // 2),
                         end=self.kernel_length // 2,
                         step=1,
                         dtype=torch.float32)
        melscale_filterbank = self.get_impuse_response(t, center_freqs, fwhms)

        return melscale_filterbank


class SincFilterBank(nn.Module):
    def __init__(self, low_freq=10, min_band_freq=10, filter_num=256,
                 kernel_len=501, sample_rate=24000):
        super(SincFilterBank, self).__init__()
        self.low_freq = low_freq
        self.min_band_freq = min_band_freq
        self.filter_num = filter_num
        self.kernel_len = kernel_len
        if self.kernel_len % 2 == 0:
            self.kernel_len += 1
        self.sample_rate = sample_rate
        self.init_sinc_learnable_params()
        self.init_half_hamming_window()

    def to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700.)

    def to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def init_sinc_learnable_params(self):
        high_freq = self.sample_rate / 2 - (self.low_freq + self.min_band_freq)
        mel = np.linspace(self.to_mel(self.low_freq), self.to_mel(high_freq),
                          self.filter_num + 1)
        hz = self.to_hz(mel)

        self.low_hz = nn.Parameter(torch.Tensor(hz[:-1]).view((-1, 1)))
        self.band_hz = nn.Parameter(torch.Tensor(np.diff(hz)).view((-1, 1)))

    def init_half_hamming_window(self):
        n_lin = torch.linspace(0, (self.kernel_len / 2) - 1,
                               steps=int(self.kernel_len / 2))
        self.hamming_window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_len)

    def get_sinc_filterbank(self):
        n = (self.kernel_len - 1) / 2.

        t = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

        low_freq = self.low_freq + torch.abs(self.low_hz)
        high_freq = torch.clamp(
            low_freq + self.min_band_freq + torch.abs(self.band_hz),
            self.low_freq,
            self.sample_rate / 2)

        band = (high_freq - low_freq)[:, 0]

        f_times_t_low = torch.matmul(low_freq, t)
        f_times_t_high = torch.matmul(high_freq, t)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(
            f_times_t_low)) / (t / 2)) * self.hamming_window
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        sinc_filter_bank = band_pass.view(self.filter_num, 1, self.kernel_len)

        return sinc_filter_bank

    def plot_sinc_filters(self):
        sinc_filter_bank = self.get_sinc_filterbank()
        sinc_filter_bank = sinc_filter_bank.detach().numpy()
        sinc_filter_bank = np.squeeze(sinc_filter_bank)
        plt.plot(sinc_filter_bank[0, :])
        plt.plot(sinc_filter_bank[100, :])
        plt.plot(sinc_filter_bank[200, :])
        plt.show()


class SynPeriodicFilterBank(nn.Module):
    def __init__(self,
                 n_fft=2048,
                 kernel_length=1025,
                 filter_num=256,
                 sample_rate=24000,
                 min_freq=1,
                 freq_init_method='melscale'):
        '''
        all essential parameter constructing mel-scale filter bank, we follow LEAF paper to construct mel-scale
        filter bank
        :param n_fft: n_fft
        :param kernel_length: kernel_length
        :param filter_num: the number of filters
        :param sample_rate: sampling rate
        '''
        super(SynPeriodicFilterBank, self).__init__()
        assert kernel_length % 2 == 1
        assert freq_init_method in ['linear', 'melscale']
        self.n_fft = n_fft
        self.kernel_length = kernel_length
        self.filter_num = filter_num
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = self.sample_rate // 2 - self.min_freq
        self.freq_init_method = freq_init_method
        self.slope_shift_group1 = [27.11, -11.8] #fitted curve parameter
        self.slope_shift_group2 = [27.11, -6.8]
        self.slope_shift_group3 = [27.11, -16.8]
        self.init_learnable_freqs_periodicity()

    def init_learnable_freqs_periodicity(self):
        if self.freq_init_method == 'linear':
            init_freqs = np.linspace(start=self.min_freq,
                                     stop=self.max_freq,
                                     num=self.filter_num,
                                     dtype=np.float32)
        elif self.freq_init_method == 'melscale':
            melscale_freqs = np.linspace(start=self.freq2mel(self.min_freq),
                                         stop=self.freq2mel(self.max_freq),
                                         num=self.filter_num,
                                         dtype=np.float32)
            init_freqs = self.mel2freq(melscale_freqs)

        half_freqs = self.sample_rate/4.
        fourth_freqs = self.sample_rate/8.

        self.half_freq_idx = np.sum( init_freqs < half_freqs )
        self.onefourth_freq_idx = np.sum( init_freqs < fourth_freqs )

        self.synperiodic_centerfreqs_goup1 = nn.Parameter(
            torch.Tensor(init_freqs).to(torch.float32))
        self.synperiodic_centerfreqs_goup2 = nn.Parameter(
            torch.Tensor(init_freqs).to(torch.float32))
        self.synperiodic_centerfreqs_goup3 = nn.Parameter(
            torch.Tensor(init_freqs).to(torch.float32))

        periodicity_group1 = self.init_initial_periodicity(init_freqs,
                                                           slope=
                                                           self.slope_shift_group1[
                                                               0],
                                                           shift=
                                                           self.slope_shift_group1[
                                                               1])
        periodicity_group2 = self.init_initial_periodicity(init_freqs,
                                                           slope=
                                                           self.slope_shift_group2[
                                                               0],
                                                           shift=
                                                           self.slope_shift_group2[
                                                               1])
        periodicity_group3 = self.init_initial_periodicity(init_freqs,
                                                           slope=
                                                           self.slope_shift_group3[
                                                               0],
                                                           shift=
                                                           self.slope_shift_group3[
                                                               1])

        self.synperiodic_periodicity_group1 = nn.Parameter(
            torch.Tensor(periodicity_group1).to(torch.float32))

        self.synperiodic_periodicity_group2 = nn.Parameter(
            torch.Tensor(periodicity_group2).to(torch.float32))

        self.synperiodic_periodicity_group3 = nn.Parameter(
            torch.Tensor(periodicity_group3).to(torch.float32))

    def init_initial_periodicity(self, center_freqs, slope=54.22, shift=-23.6):
        periodicity = shift + slope * np.log10(center_freqs)
        periodicity = np.maximum(1., periodicity)

        return periodicity

    def mel2freq(self, input_mel_freq):
        freq = (10 ** (input_mel_freq / 2595.) - 1.) * 700

        return freq

    def freq2mel(self, input_freq):
        mel_freq = 2595. * np.log10(1 + input_freq / 700.)

        return mel_freq

    def get_impulse_response(self, t, center_freqs, period_num):
        '''
        obtain the impulse response from pre-defined center-freqs and fwhms
        :param t: one-dim vector, indicating the time index, range from [-T, T]
        :param center_freqs: one-dim vector, center frequencis, of range [0, sample_rate/2]
        :param period_num: periodicity number, one-dim vector, integer number larger than 1
        :return: constructed filter bank
        '''
        center_freqs = center_freqs * (
                2 * math.pi) / self.sample_rate  # [0, pi]
        periodicity = 1. / center_freqs
        fwhms = period_num * periodicity/2.
        denominator = 1.0 / (np.sqrt(2.0 * math.pi) * fwhms)
        gaussian = torch.exp(
            torch.tensordot((1.0 / (2. * fwhms ** 2)).view((-1, 1)),
                            (-t ** 2).view((1, -1)),
                            dims=([1], [0])))

        center_freqs = center_freqs.to(torch.complex64)
        t = t.to(torch.complex64)
        sinusoid = torch.exp(1j * torch.tensordot(center_freqs.view((-1, 1)),
                                                  t.view((1, -1)),
                                                  dims=([1], [0])))
        denominator = denominator.to(torch.complex64)
        denominator = torch.unsqueeze(denominator, dim=1)
        gaussian = gaussian.to(torch.complex64)

        return denominator * sinusoid * gaussian

    def get_melscale_filterbank(self):
        center_freqs, fwhms = self.get_centerfreqs_fwhms()
        t = torch.arange(start=-(self.kernel_length // 2),
                         end=self.kernel_length // 2,
                         step=1,
                         dtype=torch.float32)
        melscale_filterbank = self.get_impuse_response(t, center_freqs, fwhms)

        return melscale_filterbank

    def get_synperiodic_filterbank_groups(self):
        t = torch.arange(start=-(self.kernel_length - 1) // 2,
                         end=(self.kernel_length - 1) // 2 + 1,
                         step=1,
                         dtype=torch.float32)
        synperiod_filterbank_group1 = self.get_impulse_response(t=t,
                                                                center_freqs=self.synperiodic_centerfreqs_goup1,
                                                                period_num=self.synperiodic_periodicity_group1)

        synperiod_filterbank_group2 = self.get_impulse_response(t=t,
                                                                center_freqs=self.synperiodic_centerfreqs_goup2,
                                                                period_num=self.synperiodic_periodicity_group2)

        synperiod_filterbank_group3 = self.get_impulse_response(t=t,
                                                                center_freqs=self.synperiodic_centerfreqs_goup3,
                                                                period_num=self.synperiodic_periodicity_group3)

        return synperiod_filterbank_group1, synperiod_filterbank_group2, synperiod_filterbank_group3

    def get_filter_bank(self):
        filterbank_group1, filterbank_group2, filterbank_group3 = self.get_synperiodic_filterbank_groups()
        filterbank_group1_list = [filterbank_group1,
                                  filterbank_group1[0:self.half_freq_idx,:],
                                  filterbank_group1[0:self.onefourth_freq_idx]]

        filterbank_group2_list = [filterbank_group2,
                                  filterbank_group2[0:self.half_freq_idx, :],
                                  filterbank_group2[0:self.onefourth_freq_idx]]

        filterbank_group3_list = [filterbank_group3,
                                  filterbank_group3[0:self.half_freq_idx, :],
                                  filterbank_group3[0:self.onefourth_freq_idx]]

        return filterbank_group1_list, filterbank_group2_list, filterbank_group3_list


class FilterBank(nn.Module):
    def __init__(self, n_fft=2048,
                 kernel_length=1025,
                 filter_num=256,
                 sample_rate = 24000,
                 min_freq = 10,
                 window='hann',
                 filter_type='linear',
                 filterbank_type='synperiodic'):
        '''
        all essential parameters for constructing filter bank
        :param n_fft: n_fft
        :param kernel_length: kernel_length, an odd number
        :param filter_num: number of filters
        :param window: the type of window to add
        :param filter_type: the type of filter bank to construct
        '''
        super(FilterBank, self).__init__()
        self.n_fft = n_fft
        self.kernel_length = kernel_length
        self.filter_num = filter_num
        self.window = window
        self.filter_type = filter_type
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.filterbank_type = filterbank_type
        assert self.filterbank_type in ['synperiodic', 'melscale','sincfilter']
        self.get_filterbank_constructor()


    def get_filterbank_constructor(self):
        if self.filterbank_type == 'synperiodic':
            self.filterbank_constuct = SynPeriodicFilterBank(n_fft=self.n_fft,
                                                             kernel_length=self.kernel_length,
                                                             filter_num=self.filter_num,
                                                             sample_rate=self.sample_rate,
                                                             min_freq=self.min_freq,
                                                             freq_init_method=self.filter_type)
        else:
            NotImplemented

    def obtain_filter_bank(self):
        return self.filterbank_constuct.get_filter_bank()

def main():
    # melscale_constructor = MelScaleFilterBank(n_fft=2048,
    #                                           kernel_length=1025,
    #                                           filter_num=256,
    #                                           sample_rate=24000)
    # melscale_filterbank = melscale_constructor.get_melscale_filterbank()
    # melscale_filterbank = melscale_filterbank.detach().numpy()
    # melscale_filterbank = np.real( melscale_filterbank )

    # plt.plot(melscale_filterbank[0,:])
    # plt.plot(melscale_filterbank[1,:])
    # plt.plot(melscale_filterbank[2,:])
    # plt.plot(melscale_filterbank[3,:])
    # plt.plot(melscale_filterbank[50, :])
    # plt.show()

    synperiod_constructor = SynPeriodicFilterBank()
    synperiod_filterbankg1, synperiod_filterbankg2, synperiod_filterbankg3 = synperiod_constructor.get_synperiodic_filterbank_groups()
    synperiod_filterbankg1 = synperiod_filterbankg1.detach().numpy()
    synperiod_filterbankg2 = synperiod_filterbankg2.detach().numpy()
    synperiod_filterbankg3 = synperiod_filterbankg3.detach().numpy()
    synperiod_filterbankg1 = np.real(synperiod_filterbankg1)
    synperiod_filterbankg2 = np.real(synperiod_filterbankg2)
    synperiod_filterbankg3 = np.real(synperiod_filterbankg3)
    plt.plot(synperiod_filterbankg1[50, :])
    plt.show()
    plt.plot(synperiod_filterbankg2[50, :])
    plt.show()
    plt.plot(synperiod_filterbankg3[50, :])
    # plt.show()
    # plt.savefig('/root/test.jpg')
    # plt.savefig('/home/zengkehan/voice/synperiodic.png')
    plt.show()


if __name__ == '__main__':
    main()
