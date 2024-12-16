import wave
import numpy as np
import matplotlib.pyplot as plt

baseFreq = 440
baseAmp = 0.5
vibratoCents = 800
vibratoDB = 0 # 6db maxxes out the audio channels
vibratoFreq = 196

sampleRate = 44000
timeFrame = 5

t = np.linspace(0, timeFrame, timeFrame*sampleRate)

w = 2*np.pi*baseFreq
wVibrato = 2*np.pi*vibratoFreq

vRatio = np.power(10, vibratoDB*np.sin(wVibrato*t)/20)
fRatio = np.power(2, vibratoCents*np.sin(wVibrato*t)/1200)
shift = np.zeros(timeFrame*sampleRate)
for time in range(1, t.shape[0]):
    shift[time] = w*(fRatio[time-1]-fRatio[time])*t[time]+shift[time-1]

data = baseAmp*vRatio*np.sin(w*fRatio*t+shift)
# data = baseAmp*vRatio*np.sin(w*t+(np.power(2, vibratoCents/1200)-1)*w/wVibrato*np.sin(wVibrato*t))

freqDomain = np.split(np.abs(np.fft.ifft(data)), 2)[0]
plt.plot(np.linspace(0, sampleRate/2, freqDomain.shape[0]), freqDomain)
plt.show()

# plt.plot(t, data)
# plt.show()

left_channel = data
right_channel = data
audio = np.array([left_channel, right_channel]).T
audio = (audio * (2 ** 15 - 1)).astype("<h")

with wave.open("sound1.wav", "w") as f:
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(sampleRate)
    f.writeframes(audio.tobytes())