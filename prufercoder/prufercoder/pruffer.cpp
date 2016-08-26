//
//  pruffer.cpp
//  prufercoder
//
//  Created by Юрий Кравченко on 28/07/16.
//  Copyright © 2016 Юрий Кравченко. All rights reserved.
//

#include "pruffer.h"
#include <vector>

using namespace std;


//hanging tree with dfs
void hangTree(int u, const vector<vector<int>> &graph, vector<int> &parent) {
    
    for (int i = 0; i < graph[u].size(); ++i) {
        int current = graph[u][i];
        
        if (parent[u] != current) {
            parent[current] = u;
            hangTree(current, graph, parent);
        }
    }
}

//linear time pruffer encouding
vector<int> getPrufferCode(const vector<vector<int>> &graph) {
    
    vector<int> parent(graph.size());
    parent[graph.size() - 1] = -1;
    
    hangTree((int)graph.size() - 1, graph, parent);
    
    int current, pointer = -1;
    
    vector<int> degree(graph.size());
    
    for (int i = 0; i < graph.size(); ++i) {
        degree[i] = (int)graph[i].size();
        
        if (degree[i] == 1 && pointer == -1) {
            pointer = i;
        }
    }
    
    vector<int> prufferCode;
    current = pointer;
    
    for (int i = 0; i < graph.size() - 2; ++i) {
        
        int next = parent[current];
        prufferCode.push_back(next);
        --degree[next];
        
        if (degree[next] == 1 && next < pointer) {
            current = next;
        }
        
        else {
            while (pointer < graph.size()) {
                ++pointer;
                
                if (degree[pointer] == 1) {
                    current = pointer;
                    break;
                }
            }
        }
    }
    return prufferCode;
}

//linear time pruffer decoding
vector<vector<int>> getTree(const vector<int> &prufferCode) {
    const int graphSize = (int) prufferCode.size() + 2;
    
    vector<int> degree(graphSize);
    
    for (int i = 0; i < graphSize; ++i) {
        degree[i] = 1;
    }
    for (int i = 0; i < prufferCode.size(); ++i) {
        ++degree[prufferCode[i]];
    }
    
    int current, pointer;
    
    for (int i = 0; i < graphSize; ++i) {
        if (degree[i] == 1) {
            pointer = i;
            break;
        }
    }
    
    vector<vector<int>> graph(graphSize);
    
    current = pointer;
    
    for (int i = 0; i < prufferCode.size(); ++i) {
        int next = prufferCode[i];
        
        graph[next].push_back(current);
        graph[current].push_back(next);
        
        --degree[current];
        --degree[next];
        
        if (degree[next] == 1 && next < pointer) {
            current = next;
        }
        
        else {
            while (pointer < graph.size()) {
                ++pointer;
                
                if (degree[pointer] == 1) {
                    current = pointer;
                    break;
                }
            }
        }
    }
    
    for (int i = 0; i < graphSize; ++i) {
        
        if (degree[i] == 1) {
            
            graph[i].push_back(graphSize - 1);
            graph[graphSize - 1].push_back(i);
            
            break;
        }
    }
    return graph;
}
    