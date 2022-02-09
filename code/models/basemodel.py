from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from keras.utils.vis_utils import plot_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from utils.miss import set_seed
from utils.callbacks import get_model_checkpoint_cb_on_min_val_loss
from utils.graphs import plot_model_performance,plot_ROC_AUC

#Base Model
class MODEL:
    
    # 1 CONV (32,(3,3),"SAME"),MAXPOOLING,
    # 1 CONV (64,(3,3),"SAME"),MAXPOOLING,    
    # dropout layer
    # Fully Connected Layer,Softmax Layer
    @staticmethod
    def BASELINE_CNN(inputshape):

      model = \
      keras.Sequential( name = "BASELINE_CNN", layers= \
          [
            keras.layers.Conv2D (16, (3, 3), padding="same", \
                                        input_shape=inputshape, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D (32, (3, 3), padding="same", activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),           
            keras.layers.Flatten(),   
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64,activation='relu'),            
            keras.layers.Dense(1, activation='sigmoid')
          ])
      # Setting Model JSON file and weight filte name 
      
      return model


def call_model(train_data_dir,validation_data_dir, \
               model_name, \
               inputshape, \
               optimizer_obj=None,batch_size = 32, \
               num_of_training_sample= 339,num_of_validation_sample= 146,\
               iterations=50):
    
    NUM_EPOCHS = iterations
    set_seed()
    model = model_name(inputshape)
    img_width, img_height = inputshape[0],inputshape[1] 
    
    if optimizer_obj==None:
        optimizer_obj = keras.optimizers.SGD(learning_rate = 0.01)
        
    model.compile(optimizer=optimizer_obj, loss='binary_crossentropy', \
                metrics=['accuracy'])
    
    print(model.summary())
    #Plot Model
    plots = plot_model(model,show_shapes=True,expand_nested=True)
    display(plots)
    
    # Save Model and its weight
    model_file_name = model.name+".json"
    model_wt_file_name_a = model.name+"_a_"+".hdf5"
    
    # Saving model in json file
    model_json = model.to_json()
    with open(model_file_name, "w") as json_file:
      json_file.write(model_json)
    
    #Rescale and data augmentation for training images
    train_datagen = ImageDataGenerator(
                        rescale=1. / 255,
                        shear_range=0.2,
                        zoom_range=[0.8,1.2],
                        rotation_range=30,
                        horizontal_flip=True,
                        vertical_flip = True)
    
    #Rescaling for validation/test images
    validate_datagen = ImageDataGenerator(rescale=1. / 255)
    
    #Get Iterators
    train_generator = train_datagen.flow_from_directory(
                                    train_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,
                                    class_mode='binary')
    
    validation_generator = validate_datagen.flow_from_directory(
                                    validation_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,shuffle = False,
                                    class_mode='binary')
    
    
    history = model.fit(
                        train_generator,
                        steps_per_epoch=num_of_training_sample // batch_size,
                        epochs=NUM_EPOCHS,
                        validation_data=validation_generator,
                        validation_steps=num_of_validation_sample // batch_size,
                        callbacks = get_model_checkpoint_cb_on_min_val_loss(model_wt_file_name_a))
    
    plot_model_performance(model.name,history) 
    
    trained_model = model_name(inputshape)
    trained_model.load_weights(model_wt_file_name_a) 
    trained_model.compile(optimizer=optimizer_obj, loss='binary_crossentropy', \
                metrics=['accuracy'])
    
    
    STEP_SIZE_TEST = validation_generator.n//validation_generator.batch_size
    validation_generator.reset()
    predicted_val = trained_model.predict(validation_generator,verbose = 1)
    fpr, tpr, _ = roc_curve(validation_generator.classes, predicted_val)
    roc_auc = auc(fpr, tpr)
    plot_ROC_AUC(fpr, tpr, roc_auc)      