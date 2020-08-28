from output_generator import OutputGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class MnistResult(OutputGenerator):
    
    def show_accuracy(self, predictions, labels):
        print(classification_report(predictions, labels))
        
    def show_cost(self, train_cost_list, test_cost_list):
        plt.plot(range(len(train_cost_list)), [m for m in train_cost_list], label = 'Train Cost')
        plt.plot(range(len(test_cost_list)), [m for m in test_cost_list], label = 'Test Cost')

        plt.legend(loc = 'best')

        plt.show()