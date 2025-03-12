# scrpt to generate a visual overview of the mood impact of the gestures on the circumplex model of emotion

import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from PIL import Image

background_image_path = os.path.join(os.path.dirname(__file__), "circumplex_model_of_emotion.png")
background_image = Image.open(background_image_path)

# dictionary can be taken from json file in moods/cat_characters directory for the corresponding cat character
gesture_vectors = {
    "Thumb_Up": (0.3, -0.1),
    "Thumb_Down": (-0.3, -0.1),
    "Closed_Fist": (-0.4, 0.2),
    "Open_Palm": (0.2, 0.4),
    "middle_finger": (-0.6, 0.6),
    "Victory": (0.4, 0.3),
    "Pointing_Up": (-0.1, 0.4),
    "Wave": (0.5, -0.3),
}

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("Valence", fontsize=12)
ax.set_ylabel("Arousal", fontsize=12)

ax.imshow(background_image, extent=[-1, 1, -1, 1], aspect='auto', alpha=0.7)

for gesture, (valence, arousal) in gesture_vectors.items():
    ax.quiver(0, 0, valence, arousal, angles='xy', scale_units='xy', scale=1, 
              color='blue', alpha=0.7, width=0.01)
    
    ax.text(valence + 0.05, arousal + 0.05, gesture, fontsize=10, 
            ha='center', va='center', color='blue',
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")])

ax.set_title("Gesture Vectors in Circumplex Model of Emotion", fontsize=14)
plt.grid(False)
plt.show()