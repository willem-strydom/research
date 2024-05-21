import numpy as np

import numpy as np

data = np.genfromtxt('access_measurements_real.csv', delimiter=',')
print(np.mean(data),(112*2 + 16)/128)