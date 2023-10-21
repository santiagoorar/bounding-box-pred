import matplotlib.pyplot as plt



def plot_metrics(df, general_title, center = 0.7):
    """ This function takes as input a dataframe with the values that are the 
        results of the train function """
    
    fig, ax = plt.subplots(1,2)

    ax[0].plot(df["Epoch"], df["Train IoU"], label = "Train IoU", color = "blue")
    ax[0].plot(df["Epoch"], df["Validation IoU"], label = "Validation IoU", color = "red")

    ax[0].set_xlabel("Number of Epochs")
    ax[0].set_ylabel("Metric (IoU)")
    ax[0].set_title("IoU across the epochs in tr/val sets ")


    ax[1].plot(df["Epoch"], df["Train Loss"], label = "Train Loss", color = "blue")
    ax[1].plot(df["Epoch"], df["Validation Loss"], label = "Validation IoU", color = "red")

    ax[1].set_xlabel("Number of Epochs")
    ax[1].set_ylabel("Metric (MSE)")
    ax[1].set_title("MSE across the epochs in tr/val sets ")

    
    ax[0].legend()
    ax[1].legend()
    
    fig.suptitle(general_title, x = center)
    plt.subplots_adjust(bottom=0.05, right=1.3, top=0.8) # adjust margins
    plt.show()    