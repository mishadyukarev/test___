import time
import pandas as pd
import ai_essay_prediction_misha.pipeline.entities_data as e
from IPython.core.display_functions import display


def timed(function):
    def wrapper(self, x: pd.DataFrame):
        #print(self.__class__.__str__(self))
        #print(self.__class__.__name__)
        display(self)


        before = time.time()
        x = function(self, x)
        after = time.time()

        spent_time = after - before
        e.spend_time_for_transform_dic[self.__class__.__name__] = spent_time

        print(f'{spent_time} - spent time')
        display(x.head(3))
        #print('_'*100)
        print('\n')




        return x

    return wrapper
