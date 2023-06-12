import matplotlib.pyplot as plt

def plot_two_lines(y_values1, y_values2, name1, name2):
    # Create a new figure
    fig, ax = plt.subplots()
    
    # Plot the first line
    ax.plot(range(len(y_values1)), y_values1, label=name1)
    
    # Plot the second line
    ax.plot(range(len(y_values2)), y_values2, label=name2)
    
    # Add a legend
    ax.legend()
    
    # Set the x and y axis labels
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Values')
    
    # Set the plot title
    ax.set_title(name1)
    
    # Show the plot
    plt.show()

with open("log_autoTrain.txt", "r") as f:
    for line in f:
        data_dict = eval(line.strip())
        for eachKey in data_dict.keys():
            if "val" in eachKey:
                continue
            name1 = eachKey
            name2 = "val_"+eachKey
            plot_two_lines(data_dict[name1],data_dict[name2],name1,name2)
        f1 = []
        val_f1 = []
        for each in range(len(data_dict["precision"])):
            f1.append(2*data_dict["precision"][each]*data_dict["recall"][each]/(data_dict["precision"][each]+data_dict["recall"][each]))
            val_f1.append(2*data_dict["val_precision"][each]*data_dict["val_recall"][each]/(data_dict["val_precision"][each]+data_dict["val_recall"][each]))
        plot_two_lines(f1,val_f1,"F1","val_F1")