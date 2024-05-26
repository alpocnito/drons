# Inspiration
# https://www.scicoding.com/introduction-to-wavelet-transform-using-python/
# https://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf

from PIL import Image
from os import listdir
from os.path import isfile, join
import time
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.signal
import wavio
from numpy.fft import rfft, rfftfreq


def load_data(file="data/LmicX-14-04-33.wav"):
    """Выгрузка аудио из файла."""
    w = wavio.read(file)
    # print(f'{len(w.data) = }')
    # print(f'{w.rate = }')
    # print(f'{w.sampwidth = }')
    F = w.rate
    w.data = w.data.ravel().astype(np.int64)
    return F, w.data


def wavelet(signal, st, en, F, title="Wavelet", wavelet_t='cgau6', save=True, fft=True):
    """Wavelet-анализ и построение графиков."""

    def plot_wavelet(ax):
        pcm = ax.pcolormesh(t, frequencies, np.abs(coefficients))
        ax.set_yscale("log")
        ax.set_xlabel("Time, s")
        ax.set_ylabel("Frequency, Hz")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax)

    st_time = time.time()

    x = signal[int(st*F) : int(en*F)]
    t = np.linspace(st, en, len(x))
    widths = np.geomspace(1, 1024, num=100)
    coefficients, frequencies = pywt.cwt(x, widths, sampling_period=1.0/F, wavelet=wavelet_t)

    print(f"Вейвлет: {time.time() - st_time:.1f} сек")

    figsize = (80, 6)
    if fft:
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        yf = rfft(x)
        xf = rfftfreq(len(x), 1.0/F)
        axs[1].semilogx(xf, np.abs(yf))
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_title("FFT")
        plot_wavelet(axs[0])
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_wavelet(ax)

    plt.tight_layout()
    if save:
        plt.savefig('img/' + title)
    plt.show()
    #plt.close('all')
    return x


def filter_highpass(F, signal, filter_cutoff=0.02):
    b = scipy.signal.firwin(201, cutoff=F*filter_cutoff, fs=F, pass_zero='highpass')
    return scipy.signal.lfilter(b, [1.0], signal)
    # w, h = scipy.signal.freqz(b, fs=F)
    # plt.title('Digital filter frequency response')
    # plt.plot(w, 20*np.log10(np.abs(h)))
    # plt.title('Digital filter frequency response')
    # plt.ylabel('Amplitude Response [dB]')
    # plt.xlabel('Frequency, Hz')
    # plt.grid()
    # plt.show()

def imshow(img_files):
    imgs = [Image.open(img_file) for img_file in img_files]
    n = len(imgs)
    plt.figure(figsize=(8*n, 15), dpi=300)
    if n > 1:
        for i, img in enumerate(imgs):
            plt.subplot(1, n, 1+i)
            plt.imshow(img)
            plt.axis('off')
    else:
        plt.imshow(imgs[0])
        plt.axis('off')
    plt.tight_layout()

def print_imgs():
    PATH = './img/'
    imshow(list(PATH + f for f in listdir(PATH) if isfile(join(PATH, f))))
