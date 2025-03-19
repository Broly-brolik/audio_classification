from moviepy.editor import AudioFileClip
import speech_recognition as sr

# Paths for the first audio file
m4a_path = "/Users/rezajabbir/Documents/HEP/24P/project_msavi/Antara_audio.m4a"
wav_path = "/Users/rezajabbir/Documents/HEP/24P/project_msavi/Antara_audio.wav"
txt_path = "/Users/rezajabbir/Documents/HEP/24P/project_msavi/Antara_audio.txt"

# Paths for the second audio file
m4a_path2 = "/Users/rezajabbir/Documents/HEP/24P/project_msavi/trajet.m4a"
wav_path2 = "/Users/rezajabbir/Documents/HEP/24P/project_msavi/trajet.wav"

# Step 1: Convert M4A to WAV using moviepy for the first audio file
audio_clip = AudioFileClip(m4a_path)
audio_clip.write_audiofile(wav_path)

# Step 1: Convert M4A to WAV using moviepy for the second audio file
audio_clip2 = AudioFileClip(m4a_path2)
audio_clip2.write_audiofile(wav_path2)

# Step 2: Transcribe the first audio file using speech_recognition
# Initialize recognizer
recognizer = sr.Recognizer()

# Convert the first audio file to audio data
with sr.AudioFile(wav_path) as source:
    audio_data = recognizer.record(source)

# Perform speech to text and save to a file
try:
    text = recognizer.recognize_google(audio_data, language='fr-FR')
    print("Transcribed Text from Antara_audio.m4a:")
    print(text)
    
    # Write the transcribed text to a file
    with open(txt_path, "w") as text_file:
        text_file.write(text)
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio from Antara_audio.m4a.")
except sr.RequestError:
    print("Sorry, my speech service is down.")

# Step 3: Transcribe the second audio file using speech_recognition
# Convert the second audio file to audio data
with sr.AudioFile(wav_path2) as source:
    audio_data2 = recognizer.record(source)

# Perform speech to text and save to a file
try:
    text2 = recognizer.recognize_google(audio_data2, language='fr-FR')
    print("Transcribed Text from trajet.m4a:")
    print(text2)
    
    # Optionally, write the transcribed text of the second audio to a file
    txt_path2 = "/Users/rezajabbir/Documents/HEP/24P/project_msavi/trajet.txt"
    with open(txt_path2, "w") as text_file2:
        text_file2.write(text2)
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio from trajet.m4a.")
except sr.RequestError:
    print("Sorry, my speech service is down.")
