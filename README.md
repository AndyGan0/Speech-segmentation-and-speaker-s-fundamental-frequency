# Speech-segmentation-and-speaker-s-fundamental-frequency
This project was developed during my university course "SPEECH AND AUDIO PROCESSING".<br>
<br>
File "_1_2_Import.py" contains the necessary methods for importing the data, analyzing audio files into mel-spectrogram, ploting the spectrogram and applying median filter.<br>
The data is split into foreground audio and background audio.<br>

Files starting with "_1_3_" are training different models for classifying the audio frames based on their label (foreground or background).<br>
Feed forward neural networks, least squares, RNN and SVM have been used. After testing all algorithms, svm was chosen. The file "svm_model.pkl" contains the trained svm.<br><br>

File "_1_5_Foreground_Playback.py" loads the svm, gets an input file and classifies each audio frame into foreground and background.<br>
After that, the output is created, which consists only from the foreground audio frames.<br><br>

File "_2_fundamental_freq.py" uses auto-correlation function to find the fundamental frequency of the speaker.<br>
Background-foreground frame classification from previous files has been used to improve the results.<br>
Additionally, a low pass filter is applied to only keep frequencies in the desired frequency range.<br>
