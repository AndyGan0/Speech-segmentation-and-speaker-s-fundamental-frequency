import numpy as np
import matplotlib.pylab as plt

from glob import glob

import librosa 
import librosa.display





def importDataStandard(ForegroundAudioPath: str = "Dataset/Training/Foreground/clean_000000/", BackgroundAudioPath: str = "Dataset/Training/Background/Myrtle/"):

    ForegroundAudioFiles = glob(ForegroundAudioPath + "*")
    ForegroundAudioFiles = [file.replace('\\', '/') for file in ForegroundAudioFiles]
    BackgroundAudioFiles = glob(BackgroundAudioPath + "*")
    BackgroundAudioFiles = [file.replace('\\', '/') for file in BackgroundAudioFiles]
    
    #   adding all files and their labels into lists
    AllFiles = []
    AllFilesLabels = []
    AllFiles.extend(ForegroundAudioFiles)     
    AllFiles.extend(BackgroundAudioFiles)    
    AllFilesLabels.extend([1] * len(ForegroundAudioFiles))   
    AllFilesLabels.extend([0] * len(BackgroundAudioFiles))

    #   permutating the lists
    permutation = np.random.permutation(len(AllFiles))
    AllFiles = np.array(AllFiles)[permutation]
    AllFilesLabels = np.array(AllFilesLabels)[permutation]

    window_size = 1024
    hop_size = int(window_size / 2)
    n_mels = 80

    #   y, SampleRate = librosa.load(ForegroundAudioFiles[0])
    #   firstWindow = y[0:window_size] * np.hamming(window_size)
    #   plotSignal(firstWindow, ForegroundAudioFiles[0])

    AllData = []
    AllDataLabels = []

    for file, label in zip(AllFiles, AllFilesLabels):
        #   Making the spectrogram from the file
        LoadedFile, SampleRate = librosa.load(file)
        mel_spectrogram = librosa.feature.melspectrogram(y=LoadedFile,
                                                        sr=SampleRate,
                                                        hop_length = hop_size,
                                                        n_fft=window_size, 
                                                        n_mels=n_mels, 
                                                        window='hamming' )
        
        #   Converting to dB
        mel_spectrogram = 10.0 * np.log10( mel_spectrogram )


        #   Append Normalized Frames in AllData
        mel_spectrogram_frames = mel_spectrogram.T
        AllData.extend(mel_spectrogram_frames)
        AllDataLabels.extend([label] * len(mel_spectrogram_frames))
        
        #   plotMelSpectrogram(mel_spectrogram, SampleRate, hop_size, file)

    AllData = np.array(AllData)    
    AllDataLabels = np.array(AllDataLabels)

    #   Normalization
    AllData = AllData.T
    for i in range(len(AllData)):
        mel_list = AllData[i]
        AllData[i] = (mel_list - np.min(mel_list)) / (np.max(mel_list) - np.min(mel_list))    
    
    #   plotMelSpectrogram(AllData, SampleRate, hop_size, "All Data")
    
    return AllData.T, AllDataLabels




RNN_batch_size = 40



def importDataForRNN(ForegroundAudioPath: str = "Dataset/Training/Foreground/clean_000000/", BackgroundAudioPath: str = "Dataset/Training/Background/Myrtle/"):

    #   import the data
    AllData, AllDataLabels = importDataStandard(ForegroundAudioPath, BackgroundAudioPath)

    batch_size = RNN_batch_size

    #   if Data frames cant be devised by batch_Size, throw away the remainder frames
    num_of_batches = len(AllData) // batch_size
    AllData = AllData[ : num_of_batches * batch_size ]
    AllDataLabels = AllDataLabels[ : num_of_batches * batch_size ]

    #   Reshape the array
    new_shape_Data = (num_of_batches, batch_size, 80)
    new_shape_labels = (num_of_batches, batch_size)

    AllData = AllData.reshape(new_shape_Data)
    AllDataLabels = AllDataLabels.reshape(new_shape_labels)
    
    return AllData, AllDataLabels








def MedianFilter(Labels: np.array, WindowSize: int):

    size_before_index = WindowSize // 2
    size_after_index = WindowSize - size_before_index

    NewLabels = np.zeros_like(Labels)

    for index in range(Labels.shape[0]):

        starting_point = index - size_before_index
        ending_point = index + size_after_index
        
        if (starting_point < 0): starting_point = 0
        if (ending_point >= Labels.shape[0]): ending_point = Labels.shape[0] - 1

        count_1 = np.sum( Labels[ starting_point : ending_point ] )
        count_0 = WindowSize - count_1

        NewLabels[index] = 1 if count_1 > count_0 else 0
    
    return NewLabels







def importSingleFile(FileAudioPath):

    window_size = 1024
    hop_size = int(window_size / 2)
    n_mels = 80


    #   Making the spectrogram from the file
    LoadedFile, SampleRate = librosa.load(FileAudioPath)
    mel_spectrogram = librosa.feature.melspectrogram(y=LoadedFile,
                                                    sr=SampleRate,
                                                    hop_length = hop_size,
                                                    n_fft=window_size, 
                                                    n_mels=n_mels, 
                                                    window='hamming' )
    
    #   Converting to dB
    mel_spectrogram = 10.0 * np.log10( mel_spectrogram )


    #   Normalization
    for i in range(len(mel_spectrogram)):
        mel_list = mel_spectrogram[i]
        mel_spectrogram[i] = (mel_list - np.min(mel_list)) / (np.max(mel_list) - np.min(mel_list))    
    
    
    return mel_spectrogram.T, LoadedFile, SampleRate







def plotMelSpectrogram(mel_spectrogram, sr, hop_size, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()








def plotSignal(signal, title, xLabel = 'Sample', yLabel = 'Amplitude'):

    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.show()






