import SimpleITK as sitk
import numpy as np
import pandas
import os
import glob
import tensorflow as tf
import gc
import resource
from tensorflow import keras
from tensorflow.keras import layers
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
wandb.init(project="monitor-gpu", entity="mustafa-ms")

x_train = []
y_train = []
x_val =[]
y_val = []
luna_path = "/home/mustafa/project/LUNA16/"
positive_nodules = pandas.read_csv(luna_path + "annotations.csv")
#luna_test_path = '/home/mustafa/project/Testfiles/test/'
luna_test_path = '/home/mustafa/project/LUNA16/subset9/'
test_file_list = glob.glob(luna_test_path + '/*.mhd')



def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0,1)
    )

#mem()


def read_mhd_file(filepath):
    """Read and load volume"""
    # Read file
    dir_check = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if os.path.samefile(dir_check, luna_test_path):
        if os.path.splitext(base)[0] in positive_nodules.values:
            y_val.append(1)
        else:
            y_val.append(0)
    else:

        if os.path.splitext(base)[0] in positive_nodules.values:
            y_train.append(1)
        else:
            y_train.append(0)

    scan = sitk.ReadImage(filepath)
    scan = sitk.GetArrayFromImage(scan)
    scan = np.moveaxis(scan, 0, 2)
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = np.flip(img , axis=2)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_mhd_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    #mem()
    return volume




for file_path in test_file_list:
    #print(file_path)
    x_val.append(process_scan(file_path))
    print("x_val = ", len(x_val))



#test_path = os.path.join(luna_path, "subset9")
#test_fold_paths = [
#    os.path.join(luna_path, test_path, z)
#    for z in os.listdir(test_path) if z.endswith('.mhd')
#    ]
#print("CT scans for test path: " + str(len(test_fold_paths)))
#test_fold = np.array([process_scan(path) for path in test_fold_paths])
#print("CT scans for test: " + str(len(test_fold)))
# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.

for subsetindex in range(9):
    luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
    #luna_subset_path = '/home/mustafa/project/Testfiles/train/'
    file_list = glob.glob(luna_subset_path + '*.mhd')

    for file_path in file_list:
        #print (file_path)
        x_train.append(process_scan(file_path))
        print("xtrain = ", len(x_train))



print("y train length ", len(y_train))
print("y test length ", len(y_val))

#mem()
@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


x_train = np.array(x_train)
np.save('x_train_1_8', x_train) # save the file as "x_train_1_8.npy"
print("xtrain = ", x_train.shape)
x_val = np.array(x_val)
np.save('x_val_9', x_val) # save the file as "x_val_9.npy"
print("xval = ", x_val.shape)
y_train = np.array(y_train)
np.save('y_train_1_8', y_train) # save the file as "y_train_1_8.npy"
print("ytrain = ", y_train.shape)
y_val = np.array(y_val)
np.save('y_val_9', y_val) # save the file as "y_val_9.npy"
print("yval = ", y_val.shape)
# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


#mem()


data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")
plt.show()


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])
#mem()




def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


#mem()
# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

#mem()
# Train the model, doing validation at the end of each epoch
epochs = 5
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb, WandbCallback()],
)
#mem()



# Save model to wandb
model.save(os.path.join(wandb.run.dir, "model.h5"))

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()
for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


#mem()
'''
# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )
'''