# Module to handle displaying and saving data plots.

# Package imports:
import matplotlib.pyplot as plt

# Classes:

# Class to store a data plot. Contains fields for the numerical data and an
# optional label.
class data_plot:
    def __init__(self, data, label = None):
        self.data = data
        self.label = label

    def has_label(self):
        return False if self.label == None else True

# Public methods:

# Method to display a list of data_plot instances in a pyplot window.
def display_plot(data_list, title, xlabel):
    __add_plot(data_list, title, xlabel)
    plt.show()

# Method to save a list of data_plot instances, plotted with pyplot, in a png
# file.
def save_plot(data_list, title, xlabel, file_name):
    figure = plt.figure()
    __add_plot(data_list, title, xlabel)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(figure)

# Private methods:

# Method to add a list of data_plot instances to pyplot.
def __add_plot(data_list, title, xlabel):
    for data_plot in data_list:
        plt.plot(data_plot.data, label=data_plot.label)
    plt.title(title)
    plt.xlabel(xlabel)
    if __any_label(data_list):
        plt.legend(loc='best')

# Method to check if a data_list has a label. Will return True if any data_plot
# in data_list has a label different than None.
def __any_label(data_list):
    return any(list(map(lambda x: x.has_label(), data_list)))
