from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz



def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y


if __name__ == "__main__":

# Sample rate and desired cutoff frequencies (in Hz).
  fs = 5000.0
  lowcut = 500.0
  highcut = 1250.0


# Filter a noisy signal.
  T = 0.05
  nsamples = T * fs
  t = np.linspace(0, T, nsamples, endpoint=False)
  a = 0.02
  f0 = 600.0
  x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
  x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
  x += a * np.cos(2 * np.pi * f0 * t + .11)
  x += 0.03 * np.cos(2 * np.pi * 2000 * t)
  plt.figure(2)
  plt.clf()
  plt.plot(t, x, label='Noisy signal')

  y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
  plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
  plt.xlabel('time (seconds)')
  plt.hlines([-a, a], 0, T, linestyles='--')
  plt.grid(True)
  plt.axis('tight')
  plt.legend(loc='upper left')

  plt.show()