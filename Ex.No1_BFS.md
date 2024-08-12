# Ex.No: 1  Implementation of Breadth First Search 
### DATE:                                                                            
### REGISTER NUMBER : 
### AIM: 
To write a python program to implement Breadth first Search. 
### Algorithm:
1. Start the program
2. Create the graph by using adjacency list representation
3. Define a function bfs and take the set “visited” is empty and “queue” is empty
4. Search start with initial node and add the node to visited and queue.
5. For each neighbor node, check node is not in visited then add node to visited and queue list.
6.  Creating loop to print the visited node.
7.   Call the bfs function by passing arguments visited, graph and starting node.
8.   Stop the program.
### Program:
```
# Using a Python dictionary to act as an adjacency list
graph = {
    '5' : ['3', '7'],
    '3' : ['2', '4'],
    '7' : ['8'],
    '2' : [],
    '4' : ['8'],
    '8' : []
}

visited = []  # List for visited nodes.
queue = []    # Initialize a queue

def bfs(visited, graph, node):
    visited.append(node)  # Mark the starting node as visited
    queue.append(node)    # Enqueue the starting node

    while queue:  # Loop to visit each node
        m = queue.pop(0)  # Dequeue a node
        print(m)  # Print the node
        
        # Visit all the neighbors of the dequeued node
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)  # Mark neighbor as visited
                queue.append(neighbour)    # Enqueue the neighbor

# Driver Code
print("Following is the Breadth-First Search:")
bfs(visited, graph, '5')  # Function calling
```

### Output:

![image](https://github.com/user-attachments/assets/d75608ba-39f3-4590-9b75-5479c1f400f5)

### Result:
Thus the breadth first search order was found sucessfully.
