import time
import pandas as pd
from IPython.core.display_functions import display


def timed(function):
    def wrapper(self, x: pd.DataFrame):
        #print(self.__class__.__str__(self))
        #print(self.__class__.__name__)
        display(self)


        before = time.time()
        x = function(self, x)
        after = time.time()


        print(f'{after - before} - spent time')
        display(x.head(3))
        #print('_'*100)
        print('\n')




        return x

    return wrapper
