import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Paths
directory = '/Users/rezajabbir/Documents/HEP/24P/project_msavi'

audio_path = directory+'ESC-50-master/audio'
metadata_path = '/Users/rezajabbir/Documents/HEP/24P/project_msavi/ESC-50-master/meta/esc50.csv'
forest_audio_path = '/Users/rezajabbir/Documents/HEP/24P/project_msavi/trajet.wav'
model_save_path = '/Users/rezajabbir/Documents/HEP/24P/project_msavi/audio_classification_model.h5'

# Load metadata
metadata = pd.read_csv(metadata_path)

# Function to load and preprocess data using Mel-spectrogram
def load_data(audio_path, metadata):
    data = []
    labels = []
    for index, row in metadata.iterrows():
        file_path = os.path.join(audio_path, row["filename"])
        audio, sr = librosa.load(file_path, sr=None)
        if audio.shape[0] < 5 * sr:
            audio = np.pad(audio, int(np.ceil((5 * sr - audio.shape[0]) / 2)), mode='reflect')
        else:
            audio = audio[:5 * sr]
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300)
        spec_db = librosa.power_to_db(spec, top_db=80)
        data.append(spec_db.flatten())  # Flatten the spectrogram to use as feature vector
        labels.append(row["category"])
    return np.array(data), np.array(labels)

# Load and preprocess data
X, y = load_data(audio_path, metadata)

# Encode labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

def augment_audio(audio, sr):
    # Add Gaussian noise
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise

    # Time-stretching
    audio_stretch = librosa.effects.time_stretch(audio, rate=1.1)

    # Pitch-shifting
    audio_shift = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=4)

    return [audio, audio_noise, audio_stretch, audio_shift]

# Augment the dataset
def load_and_augment_data(audio_path, metadata):
    metadata = pd.read_csv(metadata_path)
    data = []
    labels = []
    for index, row in metadata.iterrows():
        file_path = os.path.join(audio_path, row["filename"])
        audio, sr = librosa.load(file_path, sr=None)
        augmented_audios = augment_audio(audio, sr)
        for aug_audio in augmented_audios:
            if aug_audio.shape[0] < 5 * sr:
                aug_audio = np.pad(aug_audio, int(np.ceil((5 * sr - aug_audio.shape[0]) / 2)), mode='reflect')
            else:
                aug_audio = aug_audio[:5 * sr]
            spec = librosa.feature.melspectrogram(y=aug_audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300)
            spec_db = librosa.power_to_db(spec, top_db=80)
            data.append(spec_db.flatten())
            labels.append(row["category"])
    return np.array(data), np.array(labels)

# Load and augment data
X, y = load_and_augment_data(audio_path, metadata_path)

# Encode labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_)))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# Save the model
model.save(model_save_path)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


def split_audio(file_path, segment_duration=5):
    audio, sr = librosa.load(file_path, sr=None)
    segment_length = segment_duration * sr
    segments = []
    
    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)), mode='reflect')
        segments.append(segment)
    
    return segments, sr

def preprocess_segment(segment, sr):
    spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300)
    spec_db = librosa.power_to_db(spec, top_db=80)
    return spec_db.flatten()

def classify_long_audio(file_path, segment_duration=5):
    segments, sr = split_audio(file_path, segment_duration)
    predictions = []
    for segment in segments:
        features = preprocess_segment(segment, sr).reshape(1, -1)
        prediction = model.predict(features)
        predicted_category_index = np.argmax(prediction, axis=1)
        predicted_category = le.inverse_transform(predicted_category_index)
        predictions.append(predicted_category[0])
    
    return predictions

# Classify the long audio file
predictions = classify_long_audio(forest_audio_path)
for i, prediction in enumerate(predictions):
    print(f'Segment {i + 1}: {prediction}')

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=predictions, order=np.unique(predictions))
plt.title('Class Distribution of Forest Audio Segments')
plt.xticks(rotation=90)
plt.show()
