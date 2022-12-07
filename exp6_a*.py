from queue import PriorityQueue
def a_star(v,l,goal,h):
 open_list=PriorityQueue()
 open_list.put((13,0))
 flag=0
 path=[]
 while not(open_list.empty()) and flag==0:
 	t=open_list.get()
 	curr_cost=t[0]
 	curr=t[1]
 	path.append(curr)
 	if(curr==goal):
 		flag=1
 		break
 	for i in range(v):
 		if l[curr][i]!=-1:
 			cost=l[curr][i]
 			open_list.put((curr_cost-h[curr]+cost+h[i],i))
 if flag==1:
 	print('Goal Node is found .The Path is : ',path)
 else :
 	print('Goal Node is not found')

v=int(input('Enter the number of Nodes : '))
l=[[-1 for i in range(v)] for j in range(v)]
edge=int(input('Enter no of edges : '))
for i in range(edge):
	print('Enter the start ,end , g(n) (Cost) of edge -',i+1)
	start, end, cost=[int(x) for x in input().split()]
	l[start][end]=cost
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
a_star(v,l,goal,h)