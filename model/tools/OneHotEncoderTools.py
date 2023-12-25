import numpy as np

class OneHotEncoderTools:
    @staticmethod
    def to_full(y, num_classes):
        """Созадет массив one hot encoding.

        Args:
            y: Номер правильного класса
            num_classes: Колличество возможных классов
        
        Returns:
            Список one hot encoding.
        """
        y_full = np.zeros(num_classes)
        y_full[y - 1] = 1
        
        return y_full