from queue import PriorityQueue
def best_first_search(v,l,goal,h):
  open_list=PriorityQueue()
  open_list.put((10,0))
  flag=0
  path=[]
  while not(open_list.empty()) and flag==0:
    t=open_list.get()
    curr=t[1]
    path.append(curr)
    if(curr==goal):
      flag=1
      break
    for i in range(v):
      if l[curr][i]==1:
        open_list.put((h[i],i))
  if flag==1:
   print('Goal Node is found .The Path is : ',path)
  else :
   print('Goal Node is not found')

v=int(input('Enter the number of Nodes : '))
l=[[0 for i in range(v)] for j in range(v)]
edge=int(input('Enter no of edges : '))
for i in range(edge):
  print('Enter the start ,end of edge-',i+1)
  start, end=[int(x) for x in input().split()]
  l[start][end]=1
print('Adjacency matrix : ')
for i in range(v):
  for j in range(v):
    print(l[i][j],end=' ')
  print()
h={}
for i in range(v):
  print('Enter h(n) for Node',i+1)
  x=int(input())
  h[i]=x
goal=int(input('Enter the goal node :'))
best_first_search(v,l,goal,h)