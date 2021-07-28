# var-optimization

to install:
numpy 1.14.3
scipy 1.2.1
pillow 6.1.0
sk.learn

to run:
option 1:
  open the library in an ide
  navigate to sphere
  run sphere.py
  
option 2:
  open terminal
  navigate to sphere folder
  use command: python sphere.py

to change variables:
  in the run function in sphere.py
  there's variables in the first two lines (178,179): num_comp and num_vars to change the number of components and variables respectively
    note that the dummy dataset can only handle up to 8 variables
  div in line 184: just controls what you want the kl divergence to be
   
error:
after several iterations, at some point ans2 in line 24 becomes too big for the computer to handle
you should see several iterations run however before the library crashes
