
import os
import matplotlib as mpl; mpl.use('agg')
from keras import optimizers

from models import create_model
from data_utils import DataGenerator, HistoryPlotter, make_prediction_plots, make_input_plots, calculate_losses
from keras.callbacks import ModelCheckpoint, CSVLogger

# Training options
loss_weights      = [20.0, 0.2, 0.1]                              # Weights for balancing the loss
epochs            = 50                                            # How many epochs to train
pred_batches      = 3                                             # How many batches to do predictions on
dataset           = 'light'                                       # Or 'heavy'. Which dataset to train with
data_dir          = './Data_%s/' % dataset                        # Directory where data is loaded from
model_dir         = './model_%s/' % dataset                       # Directory where all output files are saved to
pred_dir          = os.path.join(model_dir, 'predictions/')       # Where to save predictions
checkpoint_dir    = os.path.join(model_dir, 'checkpoints/')       # Where to save model checkpoints
log_path          = os.path.join(model_dir, 'training.log')       # Where to save loss history during training
history_plot_path = os.path.join(model_dir, 'loss_history.png')   # Where to plot loss history during training
descriptors       = ['Atomic_Disks', 'vdW_Spheres', 'Height_Map'] # Used for outputting information
batch_size        = 30                                            # Number of samples per batch

# Create output folder
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Define model
model = create_model(last_relu=[False, True, True], out_labels=descriptors)
optimizer = optimizers.Adam(lr=0.001, decay=1e-5)
model.compile(optimizer, 'mse', loss_weights=loss_weights)
model.summary()
    
# Setup data loading
train_gen = DataGenerator(os.path.join(data_dir, 'train/'))
val_gen   = DataGenerator(os.path.join(data_dir, 'val/'))
test_gen  = DataGenerator(os.path.join(data_dir, 'test/'))

# Setup callbacks
checkpointer = ModelCheckpoint(checkpoint_dir+'weights_{epoch:d}.h5', save_weights_only=True)
logger = CSVLogger(log_path, append=True)
plotter = HistoryPlotter(log_path, history_plot_path, descriptors)

# Resume previous epoch if exists
init_epoch = 0
model_file = None
for i in range(1, epochs+1):
    cp_file = os.path.join(checkpoint_dir, 'weights_%d.h5' % i)
    if os.path.exists(cp_file):
        init_epoch += 1
        model_file = cp_file
    else:
        break
if init_epoch > 0:
    model.load_weights(model_file)
    print('Model weights loaded from '+cp_file)
    
# Fit model
model.fit_generator(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    initial_epoch=init_epoch,
    callbacks=[checkpointer, logger, plotter]
)

# Save final weights
model.save_weights(os.path.join(model_dir, 'model.h5'))

# Show loss history
plotter.plot()

# Test model
test_loss = model.evaluate_generator(test_gen, verbose=1)
print('Losses on training set: '+str(plotter.losses[-1]))
print('Losses on validation set: '+str(plotter.val_losses[-1]))
print('Losses on test set: '+str(test_loss))

# Make predictions
for i in range(pred_batches):
    X, true = test_gen[i]
    preds = model.predict_on_batch(X)
    losses = calculate_losses(model, true, preds)
    make_prediction_plots(preds, true, losses, descriptors, pred_dir, start_ind=batch_size*i)
    make_input_plots(X, pred_dir, start_ind=batch_size*i, constant_range=False)


