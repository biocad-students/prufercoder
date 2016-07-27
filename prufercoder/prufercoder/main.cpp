//
//  main.cpp
//  prufercoder
//
//  Created by Юрий Кравченко on 26/07/16.
//  Copyright © 2016 Юрий Кравченко. All rights reserved.
//

#include <iostream>
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <MolDraw2DSVG.h>
#include <MolDraw2D.h>
#include <GraphMol/Conformer.h>
#include <fstream>

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
    
    vector<int> parent;
    parent.resize(graph.size());
    parent[graph.size() - 1] = -1;
    
    hangTree((int)graph.size() - 1, graph, parent);
    
    int current, pointer = -1;
    
    vector<int> degree;
    degree.resize(graph.size());
    
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
vector<vector<int>> getTree(vector<int> prufferCode) {
    const int graphSize = (int) prufferCode.size() + 2;
    
    vector<int> degree;
    degree.resize(graphSize);
    
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
    
    vector<vector<int>> graph;
    graph.resize(graphSize);
    
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

vector<vector<int>> graph;
vector<int> prufferCode;

void testInit() {
    graph.resize(11);
    graph[0] = {5, 9, 10};
    graph[1] = {7};
    graph[2] = {7, 10};
    graph[3] = {7};
    graph[4] = {10};
    graph[5] = {0};
    graph[6] = {7};
    graph[7] = {1, 2, 3, 6, 8};
    graph[8] = {7};
    graph[9] = {0};
    graph[10] = {0, 2, 4};
    prufferCode = {7, 7, 10, 0, 7, 7, 2, 10, 0};
}

bool testPrufferEncoding() {
    
    vector<int> result = getPrufferCode(graph);
    
    /*for (int i = 0; i < result.size(); ++i) {
        std::cout << result[i] << ' ';
    }*/
    return result == prufferCode;
}

bool testPrufferDecoding() {
    
    vector<vector<int>> tree = getTree(prufferCode);
    for (int i = 0; i < tree.size(); ++i) {
        sort(tree[i].begin(), tree[i].end());
    }
    
    /*for (int i = 0; i < tree.size(); ++i) {
        for (int j = 0; j < tree[i].size(); ++j) {
            std::cout << tree[i][j] << ' ';
        }
        std::cout << std::endl;
    }*/
    return tree == graph;
}

void test() {
    testInit();
    assert(testPrufferEncoding());
    assert(testPrufferDecoding());
}


int main(int argc, const char * argv[]) {
    
    test();
    
    
    RDKit::ROMol * bacteriopheophytin = RDKit::SmilesToMol("CCC1[C@@H](C)c2cc3[nH]c(cc4nc([C@@H](CCC(=O)OC\C=C(/C)CCC[C@H](C)CCC[C@@H](C)CCCC(C)C)[C@@H]4C)c4[C@@H](C(=O)OC)C(=O)c5c(C)c(cc1n2)[nH]c45)c(C)c3C(C)=O");
    
    /*
    //trying to get adjacency matrix
    double * res = RDKit::MolOps::getAdjacencyMatrix(*bacteriopheophytin);
    
    auto props = bacteriopheophytin->getPropList();
    
    for (auto i = props.begin(); i != props.end(); i++) {
        std::cout << *i << std::endl;
    }
    
    STRANGE_TYPE a = bacteriopheophytin->getProp<STRANGE_TYPE>("AdjacencyMatrix");
    can't compile because of unknown type for AdjacencyMatrix
    */
    
    /*
    //trying to print our molecule
    try {
        (new RDKit::MolDraw2DSVG(1000, 1000, *new std::ofstream("123.svg")))->drawMolecule(*bacteriopheophytin);
    }
    catch(RDKit::ConformerException e) {
        std::cout << e.message() << std::endl;
    }
    //throws exception because no conformations of molecule
    */
    
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}
