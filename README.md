Cacophony Index 1.0
An attempt to start analysing birdsong with minimal stuffing around.

This process has two major functions. First it runs a voice scrubber to remove potential voices from the sound files. Then a sound event detection process is run which will output a score based on the total length of sound labelled as a sound event*. 

Voice scrubbing:
The voice detection algorithm uses the py-webrtcvad module to interact with the Google WebRTC Voice Activity Detector. This marks frames likely to have voice in the sound file. These frames are then muted.The Voice activity detector has aggressiveness settings.

Sound event detector:
This is a complicated process that I copied from Kyle Mcdonalds, AudioNotebooks[https://github.com/kylemcdonald/AudioNotebooks/blob/master/Multisamples%20to%20Samples.ipynb]. The process is meant to label certain sound events above an amplitude (~0.3) and within a certain time frame. The sound events that do get labelled are meant to be those that are most alike to the waveforms associated with birdsong. This process is not without it’s problems. However these are ‘consistent problems’ meaning that our labelled birdsong is consistent at least!

I suggest playing around with the ‘demo’ version (jupyter notebooks file) to get an idea of how the whole system works and what it’s strengths/weaknesses are. Try running the whole process on a folder of sound files and look at the waveplot’s.


Packages:
-python2.7 (all others untested)
-numpy
-librosa (https://github.com/librosa/librosa)
-matplotlib
-webrtcvad (https://github.com/wiseman/py-webrtcvad)

This code was developed for Cacophony.org.nz as their first step towards analysing bird song to help NZ become predator free by 2050, or 2040 even!
