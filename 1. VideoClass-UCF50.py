#!/usr/bin/env python
# coding: utf-8

# In[1]:


folder1 = "/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/vio_video" 
folder2 = "/Users/zaarr/Desktop/POSTDOC HBKU/CODE/4. Video/nonvio_video"


# In[2]:


import os

def make_dataset(folder, extensions):
    videos = []
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            if any(fname.endswith(ext) for ext in extensions):
                path = os.path.join(root, fname)
                videos.append(path)
    return videos
#from torchvision.datasets.folder import make_dataset
#from torchvision.datasets.video_utils import VideoClips


# In[3]:


dataset1 = make_dataset(folder1, extensions=("mp4",))
dataset2 = make_dataset(folder2, extensions=("mp4",))


# In[4]:


dataset=dataset1+dataset2


# In[5]:


print(dataset)


# In[6]:


from glob import glob

vio_files = glob("vio_video/*.mp4")
# Print the list of Python files
for file in vio_files:
    print(file)


# In[7]:


nonvio_files = glob("nonvio_video/*.mp4")
# Print the list of Python files
for file in nonvio_files:
    print(file)


# In[8]:


label=[0]*len(nonvio_files)+[1]*len(vio_files)


# In[9]:


import pandas as pd
df=pd.DataFrame(zip(nonvio_files+vio_files, label), columns = ['file', 'labels'])


# In[10]:


print(df)


# In[ ]:


import shutil
import zipfile

# Set the paths of the 'NonViolence' and 'Violence' directories
non_path = 'vio_video'
vio_path = 'nonvio_video'

# Set the output zip file path
output_zip_path = 'output.zip'

# Get the file lists for 'NonViolence' and 'Violence' directories
non_files = glob(non_path + '/*')
vio_files = glob(vio_path + '/*')

# Assign labels to the files
labels = [0] * len(non_files) + [1] * len(vio_files)

# Create a new zip file
with zipfile.ZipFile(output_zip_path, 'w') as zip_file:
    # Add files from 'NonViolence' directory to the zip with labels
    for file, label in zip(non_files, labels[:len(non_files)]):
        zip_file.write(file, arcname=f'NonViolence/{label}/{os.path.basename(file)}')

    # Add files from 'Violence' directory to the zip with labels
    for file, label in zip(vio_files, labels[len(non_files):]):
        zip_file.write(file, arcname=f'Violence/{label}/{os.path.basename(file)}')


# In[12]:


# Specify the path of the zip file
zip_path = 'output.zip'

# Specify the destination directory for extraction
extract_path = 'extracted_files'

# Extract the contents of the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


# In[2]:


import os
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
from sklearn.model_selection import train_test_split
 
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


# In[3]:


seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


# In[4]:


#!wget -nc --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar
#!unrar x UCF50.rar -inul -y


# In[6]:


# Create a Matplotlib figure
plt.figure(figsize = (30, 30))
 
# Get Names of all classes in UCF50
all_classes_names = os.listdir('UCF50')
 
# Generate a random sample of images each time the cell runs
random_range = random.sample(range(len(all_classes_names)), 20)
 
# Iterating through all the random samples
for counter, random_index in enumerate(random_range, 1):
 
    # Getting Class Name using Random Index
    selected_class_Name = all_classes_names[random_index]
 
    # Getting a list of all the video files present in a Class Directory
    video_files_names_list = os.listdir(f'UCF50/{selected_class_Name}')
 
    # Randomly selecting a video file
    selected_video_file_name = random.choice(video_files_names_list)
 
    # Reading the Video File Using the Video Capture
    video_reader = cv2.VideoCapture(f'UCF50/{selected_class_Name}/{selected_video_file_name}')
     
    # Reading The First Frame of the Video File
    _, bgr_frame = video_reader.read()
 
    # Closing the VideoCapture object and releasing all resources. 
    video_reader.release()
 
    # Converting the BGR Frame to RGB Frame 
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
 
    # Adding The Class Name Text on top of the Video Frame.
    cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
     
    # Assigning the Frame to a specific position of a subplot
    plt.subplot(5, 4, counter)
    plt.imshow(rgb_frame)
    plt.axis('off')


# In[7]:


image_height, image_width = 64, 64
max_images_per_class = 8000
 
dataset_directory = "UCF50"
classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
 
model_output_size = len(classes_list)


# In[8]:


def frames_extraction(video_path):
    # Empty List declared to store video frames
    frames_list = []
     
    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)
 
    # Iterating through Video Frames
    while True:
 
        # Reading a frame from the video file 
        success, frame = video_reader.read() 
 
        # If Video frame was not successfully read then break the loop
        if not success:
            break
 
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
         
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
         
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)
     
    # Closing the VideoCapture object and releasing all resources. 
    video_reader.release()
 
    # returning the frames list 
    return frames_list


# In[9]:


def create_dataset():
 
    # Declaring Empty Lists to store the features and labels values.
    temp_features = [] 
    features = []
    labels = []
     
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
         
        # Getting the list of video files present in the specific class name directory
        files_list = os.listdir(os.path.join(dataset_directory, class_name))
 
        # Iterating through all the files present in the files list
        for file_name in files_list:
 
            # Construct the complete video path
            video_file_path = os.path.join(dataset_directory, class_name, file_name)
 
            # Calling the frame_extraction method for every video file path
            frames = frames_extraction(video_file_path)
 
            # Appending the frames to a temporary list.
            temp_features.extend(frames)
         
        # Adding randomly selected frames to the features list
        features.extend(random.sample(temp_features, max_images_per_class))
 
        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * max_images_per_class)
         
        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()
 
    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)  
 
    return features, labels


# In[10]:


features, labels = create_dataset()


# In[11]:


one_hot_encoded_labels = to_categorical(labels)


# In[12]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 'features' is your input data (features)
# 'labels' is your corresponding output data (labels)
# test_size=0.2 specifies that 20% of the data will be used for testing
# random_state=42 sets a seed value for reproducibility

# Verify the shapes of the data splits
print("Training set shapes:")
print("Features:", features_train.shape)
print("Labels:", labels_train.shape)
print("Testing set shapes:")
print("Features:", features_test.shape)
print("Labels:", labels_test.shape)


# In[13]:


#Let's create a function that will construct our model
def create_model():
 
    # We will use a Sequential model for model construction
    model = Sequential()
 
    # Defining The Model Architecture
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size, activation = 'softmax'))
 
    # Printing the models summary
    model.summary()
 
    return model
 
 # Calling the create_model method
model = create_model()
 
print("Model Created Successfully!")


# In[17]:


plot_model(model, to_file = 'model_structure_plot.png', show_shapes = True, show_layer_names = True)


# In[18]:


plot_model(model, show_shapes = True, show_layer_names = True)


# In[16]:


pip install pydot


# In[19]:


# Adding Early Stopping Callback
from keras.utils import to_categorical

# Convert labels to one-hot encoded format
labels_train_encoded = to_categorical(labels_train)

# Define and compile your model

# Start Training




early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)
 
# Adding loss, optimizer and metrics values to the model.
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
 
# Start Training
model_training_history = model.fit(x=features_train, y=labels_train_encoded, epochs=50, batch_size=4, shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback])


# In[21]:


# Convert labels to one-hot encoded format
labels_test_encoded = to_categorical(labels_test)

# Evaluate the model
model_evaluation_history = model.evaluate(features_test, labels_test_encoded)



#model_evaluation_history = model.evaluate(features_test, labels_test)


# In[23]:


date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
model_name = f'Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

# Saving your Model
model.save(model_name)


# In[24]:


def plot_metric(metric_name_1, metric_name_2, plot_name):
  # Get Metric values using metric names as identifiers
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
 
  # Constructing a range object which will be used as time 
    epochs = range(len(metric_value_1))
   
  # Plotting the Graph
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
   
  # Adding title to the plot
    plt.title(str(plot_name))
 
  # Adding legend to the plot
    plt.legend()


# In[25]:


plot_metric('loss', 'val_loss', 'Total Loss vs Total Validation Loss')


# In[26]:


plot_metric('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')


# In[27]:


def download_youtube_videos(youtube_video_url, output_directory):
    # Creating a Video object which includes useful information regarding the youtube video.
    video = pafy.new(youtube_video_url)
 
    # Getting the best available quality object for the youtube video.
    video_best = video.getbest()
 
    # Constructing the Output File Path
    output_file_path = f'{output_directory}/{video.title}.mp4'
 
    # Downloading the youtube video at the best available quality.
    video_best.download(filepath = output_file_path, quiet = True)
 
    # Returning Video Title
    return video.title


# In[28]:


def predict_on_live_video(video_file_path, output_file_path, window_size):
 
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)
 
    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)
 
    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))
 
    while True: 
 
        # Reading The Frame
        status, frame = video_reader.read() 
 
        if not status:
            break
 
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
         
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
 
        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
 
        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
 
        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:
 
            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
 
            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
 
            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
 
            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
           
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
        # Writing The Frame
        video_writer.write(frame)
 
 
        # cv2.imshow('Predicted Frames', frame)
 
        # key_pressed = cv2.waitKey(10)
 
        # if key_pressed == ord('q'):
        #     break
 
    # cv2.destroyAllWindows()
 
     
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()
    video_writer.release()


# In[ ]:




