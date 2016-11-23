
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <assert.h>
#include "tree.cuh"

__device__ node* tempData[INPUTSIZE];
__device__ node* tempData1[INPUTSIZE];
__device__ node* root1;
__device__ node* globalCurr;
__device__ node* globalCurrs[ORDER];
__device__ node* newNode;
__device__ int globalIdx;
__device__ int tempKeys[ORDER];
__device__ node* tempPointers[ORDER];
__device__ int globalPointerIdx;

__device__ void make_node(node*& new_node)
{
	new_node = (node*)malloc(sizeof(node));
	new_node->keys = (int*)malloc( (ORDER - 1) * sizeof(int) );
	new_node->pointers = (node**)malloc( ORDER * sizeof(node *) );
	new_node->is_leaf = false;
	new_node->num_keys = 0;
	new_node->parent = NULL;
	new_node->next = NULL;
}

__device__ void make_leaf(node*& new_node)
{
	make_node(new_node);
	new_node->is_leaf = true;
}

__global__ void buildLeaves(node*& root, int* input, int* result)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned noOfNodes = INPUTSIZE / ((ORDER / 2) - 1);
	node* newNode;
	if(inWholeIdx < noOfNodes)
	{
		make_leaf(newNode);
		tempData[inWholeIdx] = newNode;
		assert(tempData[inWholeIdx]);
	}
}

__global__ void buildRoot(node*& root, int* input, int* result)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(inWholeIdx == 0)
	{
		root1 = (node*)malloc(sizeof(node));
		root1->keys = (int*)malloc( (ORDER - 1) * sizeof(int) );
		root1->pointers = (node**)malloc( ORDER * sizeof(node *) );
		root1->is_leaf = false;
		root1->num_keys = 0;
		root1->parent = NULL;
		root1->next = NULL;
		root1->keys[0] = 5;
	}
}

__global__ void buildLevel(node*& root, int* input, int* result, int size, int x)
{
	node** arr;
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned noOfNodes = size / (ORDER / 2);
	if(x)
		arr = tempData1;
	else
		arr = tempData;
	if(inWholeIdx < noOfNodes)
	{
		node* newNode;
		make_node(newNode);
		newNode->keys[0] = inWholeIdx;
		arr[inWholeIdx] = newNode;
	}
}

__global__ void fillLevel(node*& root, int* input, int* result, int size, int x)
{
	node** parent;
	node** children;
	if(x)
	{
		parent = tempData1;
		children = tempData;
	}
	else
	{
		parent = tempData;
		children = tempData1;
	}
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned noOfNodes = size / (ORDER / 2);
	unsigned inNodeIdx = inWholeIdx % (ORDER / 2);
	unsigned nodeNo = inWholeIdx / (ORDER / 2 );
	if(nodeNo == noOfNodes)
	{
		nodeNo--;
		inNodeIdx = ((ORDER/2)) + inNodeIdx;
	}
	if(inWholeIdx < size)
	{
		assert(parent[nodeNo]);
		assert(parent[nodeNo]->keys);
		assert(children[inWholeIdx]);
		parent[nodeNo]->pointers[inNodeIdx] = children[inWholeIdx];
		children[inWholeIdx]->parent = parent[nodeNo];
		if(inNodeIdx < (ORDER/2) - 1 || (nodeNo == noOfNodes -1 && inWholeIdx != size - 1))
		{
			assert(children[inWholeIdx]);
			assert(children[inWholeIdx]->num_keys);
			assert(children[inWholeIdx]->keys[children[inWholeIdx]->num_keys-1]);
			parent[nodeNo]->keys[inNodeIdx] = children[inWholeIdx]->keys[children[inWholeIdx]->num_keys-1];
			assert(parent[nodeNo]->keys[inNodeIdx]);
		}
	}
	if(inNodeIdx == 0)
	{
		if(nodeNo < noOfNodes -1)
		{
			parent[nodeNo]->num_keys = (ORDER / 2) - 1;
		}
		else if(nodeNo == noOfNodes - 1)
			parent[nodeNo]->num_keys = (size % (ORDER / 2)) + (ORDER / 2) - 1;
	}
}

__global__ void fillRoot(node*& root, int* input, int* result, int size, int x)
{
	node** children;
	if(x)
	{
		children = tempData;
	}
	else
	{
		children = tempData1;
	}
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned inNodeIdx = inWholeIdx % size;
	if(inWholeIdx < size)
	{
		assert(children[inWholeIdx]);
		root1->pointers[inNodeIdx] = children[inWholeIdx];
		children[inWholeIdx]->parent = root1;
		if(inNodeIdx < size -1 )
		{
			assert(children[inWholeIdx]);
			assert(children[inWholeIdx]->num_keys);
			assert(children[inWholeIdx]->keys[children[inWholeIdx]->num_keys-1]);
			root1->keys[inWholeIdx] = children[inWholeIdx]->keys[children[inWholeIdx]->num_keys-1];
			assert(root1->keys[inNodeIdx]);
		}
	}
	if(inWholeIdx == 0)
	{
		root1->num_keys = size - 1;
	}
}

__global__ void fillLeaves(node*& root, int* input, int* result)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned noOfNodes = INPUTSIZE / ((ORDER / 2) - 1);
	unsigned inNodeIdx = inWholeIdx % ((ORDER / 2) - 1);
	unsigned nodeNo = inWholeIdx / ((ORDER / 2 ) -1);
	if(nodeNo == noOfNodes)
	{
		nodeNo--;
		inNodeIdx = ((ORDER/2) - 1) + inNodeIdx;
	}
	if(inWholeIdx < INPUTSIZE)
	{
		assert(tempData[nodeNo]);
		assert(tempData[nodeNo]->keys);
		assert(input[inWholeIdx]);
		tempData[nodeNo]->keys[inNodeIdx] = input[inWholeIdx];
	}
	if(inNodeIdx == 0)
	{
		if(nodeNo < noOfNodes -1)
		{
			tempData[nodeNo]->next = tempData[nodeNo + 1];
			tempData[nodeNo]->num_keys = ((ORDER / 2) - 1);
		}
		else if(nodeNo == noOfNodes - 1)
			tempData[nodeNo]->num_keys = (INPUTSIZE % ((ORDER / 2) - 1)) + ((ORDER / 2) - 1);
	}
}

__global__ void bulkLoad(node*& root, int* input, int* result)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned noOfNodes = INPUTSIZE / ((ORDER / 2) - 1);
	unsigned inNodeIdx = inWholeIdx % ((ORDER / 2) - 1);
	unsigned nodeNo = inWholeIdx / ((ORDER / 2 ) -1);
	if(inWholeIdx == 0)
	{
		root->keys = (int*)malloc( (ORDER - 1) * sizeof(int) );
		root->pointers = (node**)malloc( ORDER * sizeof(node *) );
		root->is_leaf = false;
		root->num_keys = 0;
		root->parent = NULL;
		root->next = NULL;
	}
	node* newNode;
	if(inNodeIdx == 0 && nodeNo < noOfNodes)
	{
		make_leaf(newNode);
		tempData[nodeNo] = newNode;
		assert(tempData[nodeNo]);
	}
	if(nodeNo == noOfNodes)
	{
		nodeNo--;
		inNodeIdx = ((ORDER/2) - 1) + inNodeIdx;
	}
	__syncthreads();
	if(inWholeIdx < INPUTSIZE && nodeNo < noOfNodes)
	{
		tempData[nodeNo]->keys[inNodeIdx] = input[inWholeIdx];
	}
}

__device__ void addKey(node* curr, node* child)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	int val = child->keys[0];
	if(contains(curr, val))
		return;
	if(inWholeIdx <= curr->num_keys)
	{
		if(inWholeIdx < curr->num_keys)
			tempKeys[inWholeIdx] = curr->keys[inWholeIdx];
		if(!curr->is_leaf)
			tempPointers[inWholeIdx] = curr->pointers[inWholeIdx];
	}
	if(inWholeIdx <= curr->num_keys)
	{
		if(inWholeIdx == 0)
		{
			if(val <= curr->keys[0])
			{
				globalIdx = 0;
			}
		}
		else if(inWholeIdx < curr->num_keys && inWholeIdx > 0)
		{
			if(curr->keys[inWholeIdx-1] < val && val <= curr->keys[inWholeIdx])
			{
				globalIdx = inWholeIdx;
			}
		}
		else if(inWholeIdx == curr->num_keys)
		{
			if(val > curr->keys[curr->num_keys - 1])
			{
				globalIdx = curr->num_keys;
			}
		}
	}
	__syncthreads();
	if(inWholeIdx >= globalIdx && inWholeIdx <= curr->num_keys)
	{
		if(inWholeIdx < curr->num_keys)
			curr->keys[inWholeIdx+1] = tempKeys[inWholeIdx];
		if(!curr->is_leaf)
			curr->pointers[inWholeIdx+1] = curr->pointers[inWholeIdx];
	}
	__syncthreads();
	if(inWholeIdx == globalIdx)
	{
		if(inWholeIdx > 0)
			curr->keys[globalIdx] = val;
		else
			curr->keys[globalIdx] = child->keys[child->num_keys]+1;
		if(!curr->is_leaf)
			curr->pointers[globalIdx];
	}
	__syncthreads();
	if(inWholeIdx == 0)
		curr->num_keys++;
}

__device__ void addKey(node* curr, int val)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(contains(curr, val))
		return;
	if(inWholeIdx < curr->num_keys)
		tempKeys[inWholeIdx] = curr->keys[inWholeIdx];
	if(inWholeIdx <= curr->num_keys)
	{
		if(inWholeIdx == 0)
		{
			if(val <= curr->keys[0])
			{
				globalIdx = 0;
			}
		}
		else if(inWholeIdx < curr->num_keys && inWholeIdx > 0)
		{
			if(curr->keys[inWholeIdx-1] < val && val <= curr->keys[inWholeIdx])
			{
				globalIdx = inWholeIdx;
			}
		}
		else if(inWholeIdx == curr->num_keys)
		{
			if(val > curr->keys[curr->num_keys - 1])
			{
				globalIdx = curr->num_keys;
			}
		}
	}
	__syncthreads();
	if(inWholeIdx >= globalIdx && inWholeIdx < curr->num_keys)
		curr->keys[inWholeIdx+1] = tempKeys[inWholeIdx];
	__syncthreads();
	if(inWholeIdx == globalIdx)
		curr->keys[globalIdx] = val;
	__syncthreads();
	if(inWholeIdx == 0)
		curr->num_keys++;
}

__global__ void insertVal(int val)
{
	node* curr = find(val);
	__syncthreads();
	assert(curr->num_keys < ORDER -1);
	if(curr->num_keys < ORDER -1)
		addKey(curr, val);
	else
		split(curr, val);
}

__device__ void split(node* curr, int val)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	node* newNodeLocal;
	if(inWholeIdx == 0)
	{
		newNode = (node*)malloc(sizeof(node));
		newNode->keys = (int*)malloc( (ORDER - 1) * sizeof(int) );
		newNode->pointers = (node**)malloc( ORDER * sizeof(node *) );
		newNode->is_leaf = curr->is_leaf;
		newNode->num_keys = ORDER/2;
		newNode->parent = curr->parent;
		newNode->next = curr->next;
		curr->num_keys = ORDER/2;
		curr->next = newNode;
		globalPointerIdx = 0;
	}
	__syncthreads();
	newNodeLocal = newNode;
	__syncthreads();
	if(inWholeIdx < (ORDER /2))
	{
		newNode->keys[inWholeIdx] = curr->keys[ORDER/2 + inWholeIdx];
	}
	if(!curr->is_leaf && inWholeIdx <= (ORDER /2))
	{
		newNode->leafs[inWholeIdx] = curr->pointers[ORDER/2 + inWholeIdx];
	}
	if(curr->parent->num_keys >= ORDER)
		split(curr, newNode);
	else
		addKey(curr->parent, newNodeLocal);
}

__device__ void split(node* curr, node* child)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	node* newNodeLocal;
	if(inWholeIdx == 0)
	{
		newNode = (node*)malloc(sizeof(node));
		newNode->keys = (int*)malloc( (ORDER - 1) * sizeof(int) );
		newNode->pointers = (node**)malloc( ORDER * sizeof(node *) );
		newNode->is_leaf = curr->is_leaf;
		newNode->num_keys = ORDER/2;
		newNode->parent = curr->parent;
		newNode->next = curr->next;
		curr->num_keys = ORDER/2;
		curr->next = newNode;
		globalPointerIdx = 0;
	}
	__syncthreads();
	newNodeLocal = newNode;
	__syncthreads();
	if(inWholeIdx < (ORDER /2))
	{
		newNode->keys[inWholeIdx] = curr->keys[ORDER/2 + inWholeIdx];
	}
	if(!curr->is_leaf && inWholeIdx <= (ORDER /2))
	{
		newNode->leafs[inWholeIdx] = curr->pointers[ORDER/2 + inWholeIdx];
	}
	if(curr->parent->num_keys >= ORDER)
		split(curr, newNode);
	else
		addKey(curr->parent, newNodeLocal);
}
__device__ int contains(node* curr, int val)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(inWholeIdx < curr->num_keys)
	{
		if(curr->keys[inWholeIdx] == val)
			globalIdx = 1;
	}
	__syncthreads();
	return globalIdx;
}

__device__ node* find(int val)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	assert(root1);
	node* curr = root1;
	assert(curr);
	assert(!curr->is_leaf);
	__syncthreads();
	while(!curr->is_leaf)
	{
		if(inWholeIdx <= curr->num_keys)
		{
			if(inWholeIdx == 0)
			{
				assert(curr->keys[0]);
				if(val <= curr->keys[0])
				{
					assert(curr->pointers[0]);
					globalCurr = curr->pointers[0];
				}
			}
			else if(inWholeIdx < curr->num_keys && inWholeIdx > 0)
			{
				assert(curr->keys[inWholeIdx-1]);
				assert(curr->keys[inWholeIdx]);
				if(curr->keys[inWholeIdx-1] < val && val <= curr->keys[inWholeIdx])
				{
					assert(curr->pointers[inWholeIdx]);
					globalCurr = curr->pointers[inWholeIdx];
				}
			}
			else if(inWholeIdx == curr->num_keys)
			{
				assert(curr->keys[curr->num_keys - 1]);
				if(val > curr->keys[curr->num_keys - 1])
				{
					assert(curr->pointers[inWholeIdx]);
					globalCurr = curr->pointers[inWholeIdx];
				}
			}
		}
		__syncthreads();
		curr = globalCurr;
		__syncthreads();
	}
	return curr;
}

__device__ node* find(int* values, int len)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned inNodeIdx = threadIdx.x;
	unsigned nodeNo = blockIdx.x;
	int val;
	if(nodeNo < len)
		val = values[nodeNo];
	assert(root1);
	node* curr = root1;
	assert(curr);
	assert(!curr->is_leaf);
	__syncthreads();
	while(!curr->is_leaf)
	{
		if(inNodeIdx <= curr->num_keys && nodeNo < len)
		{
			if(inNodeIdx == 0)
			{
				assert(curr->keys[0]);
				if(val <= curr->keys[0])
				{
					assert(curr->pointers[0]);
					globalCurrs[nodeNo] = curr->pointers[0];
				}
			}
			else if(inNodeIdx < curr->num_keys && inNodeIdx > 0)
			{
				assert(curr->keys[inNodeIdx-1]);
				assert(curr->keys[inNodeIdx]);
				if(curr->keys[inNodeIdx-1] < val && val <= curr->keys[inNodeIdx])
				{
					assert(curr->pointers[inNodeIdx]);
					globalCurrs[nodeNo] = curr->pointers[inNodeIdx];
				}
			}
			else if(inNodeIdx == curr->num_keys)
			{
				assert(curr->keys[curr->num_keys - 1]);
				if(val > curr->keys[curr->num_keys - 1])
				{
					assert(curr->pointers[inNodeIdx]);
					globalCurrs[nodeNo] = curr->pointers[inNodeIdx];
				}
			}
		}
		__syncthreads();
		assert(globalCurrs[nodeNo]);
		curr = globalCurrs[nodeNo];
		__syncthreads();
	}
	return curr;
}

__global__ void search(int val, int* result)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	node* curr = find(val);
	result[0] = 0;
	if(inWholeIdx < curr->num_keys)
	{
		if(curr->keys[inWholeIdx] == val)
			result[0] = 1;
	}
}

__global__ void search(int* vals, int* results, int len)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned inNodeIdx = threadIdx.x;
	unsigned nodeNo = blockIdx.x;
	node* curr = find(vals, len);
	if(nodeNo < len)
		results[nodeNo] = 0;
	if(nodeNo < len && inNodeIdx < curr->num_keys)
	{
		if(curr->keys[inNodeIdx] == vals[nodeNo])
			results[nodeNo] = 1;
	}
}

__global__ void test(node*& root, int* input, int* result)
{
	unsigned inWholeIdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(inWholeIdx== 0)
	{
		node* curr = root1;
		/*while(!curr->is_leaf)
		{
			curr = curr->pointers[2];
		}*/
		result[0] = root1->pointers[root1->num_keys]->keys[0];
	}
}
