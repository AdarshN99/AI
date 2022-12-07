def dfs(a, vis,source,n):
    for i in range(n):
        if (a[source][i]==1 and vis[i]==0):
            vis[i]=1
            print(i,end=' ')
            dfs(a,vis,source,n)
v=int(input())
a=[]
for i in range(v):
    b=[0]*v
    a.append(b)
e=int(input())
vis=[0]*v
for i in range(e):
    l1,l2=map(int,input().split(' '))
    a[l1][l2]=1
print(0,end=' ')
dfs(a,vis,0,v)

