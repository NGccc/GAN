import numpy as np
import cv2
def quart2Rotationvector(*nums):
  R = np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=float)
#print(RotationMatrix)
  x = float(nums[0])  #  img items: x,y,z
                      # real items :w  #
  y = float(nums[1])
  z = float(nums[2])
  w = float(nums[3])
#calculate element of matrix
  R[0][0] = np.square(w) + np.square(x) - np.square(y) - np.square(z)
  R[0][1] = 2*(x*y + w*z)
  R[0][2] = 2*(x*z - w*y)
  R[1][0] = 2*(x*y - w*z)
  R[1][1] = np.square(w) - np.square(x) + np.square(y) - np.square(z)
  R[1][2] = 2*(w*x + y*z)
  R[2][0] = 2*(x*z + w*y)
  R[2][1] = 2*(y*z - w*x)
  R[2][2] = np.square(w) - np.square(x) - np.square(y) + np.square(z)
  vector = cv2.Rodrigues(R)
  return vector[0]

if __name__ == '__main__':
  testquart = (0.477869,-0.542066,-0.522417,-0.452644)
  import pdb
  pdb.set_trace()
  res=quart2Rotationvector(testquart)
  print('??')