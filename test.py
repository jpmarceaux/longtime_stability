# append a new hello to a file in this directory
from erpe.experiment_design import *


# open the file
with open('/home/jpmarceaux/longtime_stability/hello.txt', 'a') as f:
    f.write('hello\n')

# open the file
