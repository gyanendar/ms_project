import numpy as np
import matplotlib.pyplot as plt


# Performance plot for a model depicting training/validation loss/accuracy

def plot_model_performance(name,history):
  plt.style.use("ggplot")
  plt.figure(figsize=(7,6))
                    
  epoch_ran = len(history.history["loss"])
  plt.plot(np.arange(0, epoch_ran), \
           history.history["loss"], label="train_loss")
  plt.plot(np.arange(0, epoch_ran), \
           history.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, epoch_ran), \
           history.history["accuracy"], label="train_acc")
  plt.plot(np.arange(0, epoch_ran), \
           history.history["val_accuracy"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()

  #Print best score 
  print("\nModel Performance Summary:\n")
  
  
  min_val_loss = min(history.history["val_loss"])

  epoch_index = history.history["val_loss"].index(min_val_loss)
    
  best_val_accuracy = history.history["val_accuracy"][epoch_index]

  print("Min validation loss:",min_val_loss," at EPOCH:", epoch_index+1)

  print("\nValidation Accuracy: ",best_val_accuracy)


def plot_ROC_AUC(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_images(train_dataset):    
    plt.figure(figsize=(10, 10))
    class_names = train_dataset.class_names
    
    for images, labels in train_dataset.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")    