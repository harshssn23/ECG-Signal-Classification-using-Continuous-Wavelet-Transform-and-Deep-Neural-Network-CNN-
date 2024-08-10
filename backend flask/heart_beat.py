import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def hb(image):   
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # Convert edge image to 1D signal by summing along the y-axis
    signal = np.sum(edges, axis=0)

    # Normalize the signal
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Detect peaks in the signal
    peaks, properties = find_peaks(signal, height=0.5, distance=30)  # Adjust height and distance as needed

    # Print number of detected peaks
    print(f'Detected R-peaks: {len(peaks)}')

    # Calculate heart rate
    time_interval_seconds = 4.0  # Assuming the x-axis of the image represents 10 seconds
    heart_rate = (len(peaks) / time_interval_seconds) * 60
    hr=round(heart_rate)

    print(f'Heart Rate: {hr} bpm')
    return hr

# img=cv2.imread(r'C:\Users\Harsh\OneDrive\Desktop\Sleep Pattern Recog\cnn\ecgdataset\arr\502.jpg')
# a=hb(img)
# print(a)