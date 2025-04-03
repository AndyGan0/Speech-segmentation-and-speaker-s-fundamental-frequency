import librosa
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_prominences, butter, filtfilt
from _1_5_Foreground_Playback import getForegroundAudioSignal

import pyaudio
import soundfile as sf






def lowPassFilter(signal, sample_rate, cutoff_freq=300):

    order = 3
    NyquistFreq = 0.5 * sample_rate  # Nyquist frequency

    normal_cutoff = cutoff_freq / NyquistFreq

    #   getting filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    #   Applying the filter to the signal
    FilteredSignal = filtfilt(b, a, signal)

    return FilteredSignal





#   importing the file
window_size = 1024
hop_size = int(window_size / 2)
n_mels = 80


#   Making the spectrogram from the file
filepath = "Dataset/Training/Foreground/clean_000000/07282016HFUUforum_SLASH_07-28-2016_HFUUforum_DOT_mp3_00022.flac"
file, SampleRate = getForegroundAudioSignal(filepath)
file = lowPassFilter(file, SampleRate)



#   keeping a list with the fundamental freq of all frames
fundamental_freqs_from_all_frames = []

#   for each frame calculating the fundamental frequency
last_frame_index = len(file-window_size)
for i in range(0, last_frame_index, hop_size):
    #   exctract the frame
    current_frame = file[i : i + window_size]
    #   applying hamming function
    current_frame = current_frame * np.hamming(len(current_frame))


    #   subtructing the mean
    current_frame = current_frame - np.mean(current_frame)
    correlation = np.correlate(current_frame, current_frame, mode='full')
    correlation = correlation[correlation.size // 2:]

    #   normalizing the correlation
    correlation = (correlation - np.min(correlation)) / (np.max(correlation) - np.min(correlation))    

    height_threshold = 0.7
    prominence_threshold = 0.2


    #   from the correlation values, keep only the peaks
    peaks, _ = find_peaks(correlation, height=height_threshold)
    prominences = peak_prominences(correlation, peaks)[0]

    filtered_peaks = []
    for i in range(len(peaks)):
        if prominences[i] >= prominence_threshold:
            filtered_peaks.append(peaks[i])
    peaks = filtered_peaks
    


    #   if there are no peaks then signal frequency is 0
    if len(peaks) == 0:
        continue
    else:
        first_peak_index = peaks[0] #   first_peak index == lag for the first peak
        fundamental_frequency_of_current_frame = SampleRate / first_peak_index    
        fundamental_freqs_from_all_frames.append(fundamental_frequency_of_current_frame)
    
        


fundamental_freqs_from_all_frames = np.array(fundamental_freqs_from_all_frames)


average_fundamental_freq = np.average(fundamental_freqs_from_all_frames) 
print()
print(average_fundamental_freq, " Hz")


# Plot the fundamental frequency over time
plt.plot(fundamental_freqs_from_all_frames)
plt.xlabel('Frame Index')
plt.ylabel('Frequency (Hz)')
plt.title('Fundamental Frequency Over Time')
plt.show()








