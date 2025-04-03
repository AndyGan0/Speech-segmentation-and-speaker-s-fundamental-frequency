from _1_2_Import import importSingleFile, MedianFilter

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import joblib
import pyaudio
import soundfile as sf


#   Creating the model
model = joblib.load('svm_model.pkl')




def getForegroundAudioSignal(path):

    #   Importing the data using the method in _1_2_Import.py
    MelSpectrogram, File, SampleRate = importSingleFile(path)

    #   predicting
    predictions = model.predict(MelSpectrogram)
    predictions = MedianFilter(predictions, 20)

    ForegroundAudioSignal = []
    #   removing from the file frames that are 0
    for i in range(len(predictions)):
        if predictions[i] == 1:

            #   start index of cut
            start_index = 512 * i

            #   end index of cut
            end_index = start_index + 1024
            
            if i > 0 and predictions[i - 1] == 1:
                #   if previous frame was added, then we'll have overlap
                start_index += 512       

            if end_index > len(File):
                end_index = len(File)

            #   if frame is last frame and last 2 frames are 1, then start_index will be bigger than end index
            #   we need to make sure that this doesnt happen
            if start_index < end_index:
                ForegroundAudioSignal.extend(File[start_index:end_index])


    ForegroundAudioSignal = np.array(ForegroundAudioSignal)
    return ForegroundAudioSignal,SampleRate





#   this code will execute ONLY IF the file is run
#   if the function getForegroundAudioSignal is imported into another file, the code wont run
if __name__ == "__main__":
    
    ForegroundAudioFile, SampleRate = getForegroundAudioSignal("Dataset/Final_Playback_Test/final_test.wav")

    #   saving the audio file
    output_filename = 'output_audio_After_Removing_Background.wav'
    sf.write(output_filename, ForegroundAudioFile, SampleRate)

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SampleRate,
                    output=True)

    stream.write(ForegroundAudioFile.astype(np.float32).tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()

