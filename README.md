# SignAlong: AI-Powered ASL Music Tutor

SignAlong: Learn ASL Through Music
Where Machine Learning and Computer Vision meet rhythm, gesture, and human expression.

SignAlong is an AI-powered, vision-based American Sign Language learning experience that lets users perform ASL signs in sync with music, receive feedback, and learn interactively.

Built with MediaPipe, OpenCV, and custom gesture-template recognition, SignAlong supports multiple songs (e.g., YMCA and Love You Like a Love Song) and delivers real-time performance scoring.

Features
1. ASL Karaoke Mode: Perform ASL signs synchronously to:
- YMCA (static letter signs)
- Love You Like a Love Song (multi-hand dynamic ASL signs)

Each song has:
Beginner mode (slow audio)
Normal mode
Synchronized expected-sign timeline
Real-time predicted-sign overlay
End-of-chorus summary score

2. Template-Based Gesture Recognition

SignAlong uses a user-trained template system:
Each ASL sign (letters or words) is captured via a dedicated capture_templates.py


Before performance, users can practice each ASL sign with:

Reference images

Real-time scoring

This helps beginners learn each sign before performing the song.

4. Real-Time Vision & Audio Sync

Hand landmarks from MediaPipe

Custom 63-dimensional feature vectors

k-NN template comparison

Live scoring

Real-time progress bars

Song-specific expected-sign visual cue

Optional facial-expression expansions



🕹️ How to Use
1. Install dependencies
pip install mediapipe==0.10.21 sounddevice simpleaudio opencv-python numpy

2. Capture Templates Before Playing

Run:

python capture_templates.py


Choose the song → perform each sign → templates will be saved automatically.

3. Run the Main App
python main.py

🎤 Supported Songs
1. YMCA

Signs: Y, M, C, A

Static letters

Great for beginners

2. Love You Like a Love Song (LYLLS)

Signs:

B – BABY

I – ILY

k – LIKE

a – A

l – LOVE

s – SONG

Includes dynamic & two-hand signs.
