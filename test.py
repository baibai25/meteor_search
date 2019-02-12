import glob
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model

# Load video
def load_video_as_array(videopath):
    cap = cv2.VideoCapture(videopath)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.asarray(frames)
    x = np.mean(frames, axis=0)  # compute the mean of frames over time
    return x
    
if __name__ == "__main__":
    # Load test set
    file = sorted(glob.glob('test/*'))
    X_test = []
    name = []
    for i in range(len(file)):
        #if i >= 100:
            #break
        X_test.append(load_video_as_array(file[i]))
        name.append(os.path.basename(file[i]))
    X_test = np.asarray(X_test).astype("float16")
    X_test /= 255.0
    
    # Testing and export result
    model = load_model('my_model.h5')
    y_prob = model.predict_proba(X_test)
    pd.DataFrame({'ID':name, 'prob':y_prob[:, 1]}).to_csv("submit.csv", index=False, header=False)
