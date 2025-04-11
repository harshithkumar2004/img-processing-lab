import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import imageio

# Load grayscale image
image = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found. Make sure 'image3.jpg' is in the current directory.")

kernel = np.ones((3, 3), np.uint8)
iterations = 5
evolution_frames = []
entropy_values = []

# Function to calculate entropy of pixel values
def calc_entropy(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / np.sum(hist)
    return entropy(hist_norm + 1e-10, base=2)

# Store edge persistence info
persistence_map = np.zeros_like(image, dtype=np.uint8)

# Initial image for comparison
prev_image = image.copy()

for i in range(1, iterations + 1):
    eroded = cv2.erode(prev_image, kernel, iterations=1)
    abs_diff = cv2.absdiff(prev_image, eroded)

    # Accumulate persistence (if pixel changes, it's part of an evolving edge)
    persistence_map = cv2.add(persistence_map, (abs_diff > 15).astype(np.uint8) * 20)

    # Store frame for GIF
    color_map = cv2.applyColorMap(abs_diff, cv2.COLORMAP_JET)
    evolution_frames.append(color_map)

    # Calculate entropy
    ent = calc_entropy(abs_diff)
    entropy_values.append(ent)

    prev_image = eroded.copy()

# Classify Edge Behavior based on Persistence
stable_edges = (persistence_map > 60).astype(np.uint8) * 255
transient_edges = ((persistence_map > 20) & (persistence_map <= 60)).astype(np.uint8) * 255
weak_edges = (persistence_map <= 20).astype(np.uint8) * 255

# Visualization
plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Persistence Map (Edge Life)")
plt.imshow(persistence_map, cmap='magma')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Stable Edges")
plt.imshow(stable_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Transient Edges")
plt.imshow(transient_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Weak Edges")
plt.imshow(weak_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Edge Entropy Over Iterations")
plt.plot(range(1, iterations+1), entropy_values, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Entropy (bits)")

plt.tight_layout()
plt.show()

# Save evolution as GIF
imageio.mimsave('edge_evolution.gif', evolution_frames, fps=2)
print("🧬 Edge Evolution GIF saved as 'edge_evolution.gif'")
