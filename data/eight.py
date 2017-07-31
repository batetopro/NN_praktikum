import random      
import math
import matplotlib
import matplotlib.pyplot as plt

F = [
  lambda x: 0,
  lambda x: x,
  lambda x: (10 ** 100) * x,
  lambda x: -x,
]


p = 0
N = 700
M = 2
d = 0.2

result = {k: [] for k in range(8)}

while p < 8 * N:
  x,y = random.uniform(-1, 1) * M, random.uniform(-1, 1) * M
  r = math.sqrt(x**2 + y**2)
  if r > M: continue
  
  f = [F[k](x) for k in range(4)]
  
  if y >= f[0] and y <= f[1]:
    if result[0].__len__() < N:
      result[0].append((x + d, y + d / 2))
      p+=1
  
  if y >= f[1] and x > 0:
    if result[1].__len__() < N:
      result[1].append((x + d / 2, y + d))
      p+=1
  
  if x < 0 and y >= f[3]:
    if result[2].__len__() < N:
      result[2].append((x - d / 2, y + d))
      p+=1
  
  if y >= f[0] and y <= f[3]:
    if result[3].__len__() < N:
      result[3].append((x - d, y + d / 2))
      p+=1
  
  if y <= f[0] and y >= f[1]:
    if result[4].__len__() < N:
      result[4].append((x - d, y - d / 2))
      p+=1  
  
  if x < 0 and y <= f[1]:
    if result[5].__len__() < N:
      result[5].append((x - d / 2, y - d))
      p+=1
  
  if x > 0 and y <= f[3]:
    if result[6].__len__() < N:
      result[6].append((x + d / 2, y - d))
      p+=1
  
  if y <= f[0] and y >= f[3]:
    if result[7].__len__() < N:
      result[7].append((x + d, y - d / 2))
      p+=1
      
x, y, color = [], [], []
colors = [
    '#FF0000',
    '#C71585',
    '#808000',
    '#FFD700',
    '#FF00FF',
    '#800080',
    '#483D8B',
    '#8B0000',
    "#228B22",
    "#008080"
]

r = 256 / (2*(M+d))

output = []
for c in result:
  for p in result[c]:
    output.append("%d,%d,%d" % (c, (M + p[0])*r, (M + p[1])*r))
    color.append(colors[c])
    x.append(int((d + M + p[0])*r))
    y.append(int((d + M + p[1])*r))

plt.scatter(x, y, color=color)
plt.show()

random.shuffle(output)

for p in output:
  print(p)