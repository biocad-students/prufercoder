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
#include <GraphMol/Substruct/SubstructMatch.h>
#include <GraphMol/BondIterators.h>
#include <GraphMol/AtomIterators.h>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <GraphMol/SmilesParse/SmilesWrite.h>

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
vector<vector<int>> getTree(vector<int> prufferCode) {
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

//reading rings.txt
vector<RDKit::ROMol*> getRings() {
    vector<RDKit::ROMol*> result;
    ifstream fin("data/rings.txt");
    while (!fin.eof()) {
        string mol;
        fin >> mol;
        result.push_back(RDKit::SmilesToMol(mol));
    }
    return result;
}

/*void createRings() {
    set<string> mols;
    vector<string> names = {"frequent.txt", "common.txt", "in-man.txt", "in-drugs.txt", "endogenous.txt", "in-nature.txt"};
    for (int i = 0; i < names.size(); ++i) {
        ifstream fin("data/" + names[i]);
        string s;
        while (getline(fin, s)) {
            vector<string> strings;
            boost::split(strings, s, boost::is_any_of("\t"));
            mols.insert(strings[1]);
        }
    }
    ofstream fout("data/rings.txt");
    for (auto i = mols.begin(); i != mols.end(); ++i) {
        fout << *i << endl;
    }
}*/

const int MAX_EQUAL_RINGS_COUNT = 1000;
vector<int> squeezeRings(RDKit::ROMol * mol, vector<RDKit::ROMol*> rings) {
    vector<int> atomMapping(mol->getNumAtoms(), -1);
    
    for (int i = 0; i < rings.size(); ++i) {
        
        vector<RDKit::MatchVectType> match;
        
        if (RDKit::SubstructMatch(*mol, *rings[i], match)) {
            
            for (int j = 0; j < match.size(); ++j) {
                for (int k = 0; k < match[j].size(); ++k) {
                    
                    if (atomMapping[match[j][k].second] == -1 || rings[i]->getNumAtoms() > rings[atomMapping[match[j][k].second] / MAX_EQUAL_RINGS_COUNT]->getNumAtoms()) {
                        
                        atomMapping[match[j][k].second] = i * MAX_EQUAL_RINGS_COUNT + j;
                    }
                    
                }
            }
        }
    }
    
    for (int i = 0; i < atomMapping.size(); ++i) {
        if (atomMapping[i] == -1) {
            atomMapping[i] = i;
        }
        else {
            atomMapping[i] += MAX_EQUAL_RINGS_COUNT;
        }
    }
    return atomMapping;
}

//mapping numbers from 0 to n-1
map<int, int> map0N(vector<int> &vertexes) {
    map<int, int> mapping;
    for (int i = 0; i < vertexes.size(); ++i) {
        mapping[vertexes[i]] = i;
    }
    return mapping;
}

vector<string> atomSet = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Br", "Cl", "B", "C", "N", "O", "P", "S", "F", "I", "H", "+", "-"};

int atomEncoding(string s) {
    int res = 0;
    for (int i = 0; i < atomSet.size(); ++i) {
        if (atomSet[i] == s) {
            return i;
        }
    }
    for (int i = 1; i < s.size() - 1; ++i) {
        for (int j = 0; j < atomSet.size(); ++j) {
            bool match = true;
            for (int k = 0; k < atomSet[j].size(); ++k) {
                if (s[i + k] != atomSet[j][k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                res *= atomSet.size();
                res += j;
                i += atomSet[j].size() - 1;
                break;
            }
        }
    }
    return -res;
}

string atomDecoding(int code) {
    if (code > 0) {
        return atomSet[code];
    }
    code *= -1;
    string s = "]";
    while (code > 0) {
        s += atomSet[code % atomSet.size()];
        code /= atomSet.size();
    }
    s += "[";
    reverse(s.begin(), s.end());
    return s;
}

pair<vector<vector<int>>, vector<int>> molToTree(RDKit::ROMol * mol) {
    vector<RDKit::ROMol*> rings = getRings();
    vector<int> atomMapping = squeezeRings(mol, rings);
    vector<pair<int, int>> edges;
    vector<int> vertexes;
    
    for (auto i = mol->beginBonds(); i != mol->endBonds(); ++i) {
        unsigned int u = (*i)->getBeginAtomIdx(), v = (*i)->getEndAtomIdx();
        
        vertexes.push_back(atomMapping[u]);
        vertexes.push_back(atomMapping[v]);
        
        if (atomMapping[u] == atomMapping[v]) {
            continue;
        }
        if (atomMapping[u] < atomMapping[v]) {
            edges.push_back(make_pair(atomMapping[u], atomMapping[v]));
        }
        else {
            edges.push_back(make_pair(atomMapping[v], atomMapping[u]));
        }
    }
    /*for (int i = 0; i < vertexes.size(); ++i) {
        cout << vertexes[i] << ' ';
    }
    cout << endl;
     */
    sort(edges.begin(), edges.end());
    auto lastEdge = unique(edges.begin(), edges.end());
    edges.erase(lastEdge, edges.end());
    sort(vertexes.begin(), vertexes.end());
    auto lastVertex = unique(vertexes.begin(), vertexes.end());
    vertexes.erase(lastVertex, vertexes.end());
    
    /*for (int i = 0; i < vertexes.size(); ++i) {
        cout << vertexes[i] << ' ';
    }
    cout << endl;

    for (int i = 0; i < edges.size(); ++i) {
        cout << edges[i].first << " - " << edges[i].second << endl;
    }*/
    
    map<int, int> mapping0N = map0N(vertexes);
    
    vector<vector<int>> graph(vertexes.size());
    for (int i = 0; i < edges.size(); ++i) {
        int u = mapping0N[edges[i].first], v = mapping0N[edges[i].second];
        
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    
    for (int i = 0; i < vertexes.size(); ++i) {
        if (vertexes[i] < MAX_EQUAL_RINGS_COUNT) {
            string s = RDKit::SmilesWrite::GetAtomSmiles(mol->getAtomWithIdx(vertexes[i]));
            vertexes[i] = atomEncoding(s);
        }
    }
    
    return make_pair(graph, vertexes);
}

bool testMatch(RDKit::ROMol * mol1, RDKit::ROMol * mol2) {
    RDKit::MatchVectType res;
    return RDKit::SubstructMatch(*mol1, *mol2, res);
}

void atomCodeTest() {
    string s = "Cl";
    int a = atomEncoding(s);
    assert(s == atomDecoding(a));
    
    s = "C";
    a = atomEncoding(s);
    assert(s == atomDecoding(a));
    
    s = "[NH2+]";
    a = atomEncoding(s);
    assert(s == atomDecoding(a));
    
    s = "[13CH4]";
    a = atomEncoding(s);
    assert(s == atomDecoding(a));
}

int main(int argc, const char * argv[]) {
    cerr << "in main" << endl;
    
    //createRings();
    
    //test();
    
    //atomCodeTest();
    
    //getRings();
    
    RDKit::ROMol * myMol = RDKit::SmilesToMol("C[C@H]1C[C@@H](C[NH2+]Cc2cc(Cl)c3c(c2)OCCCO3)CCO1");
    /*for (auto i = myMol->beginAtoms(); i != myMol->endAtoms(); ++i) {
        cout << RDKit::SmilesWrite::GetAtomSmiles(*i) << endl;
    }*/
    
    //cout << testMatch(bacteriopheophytin, bacteriopheophytin);
    auto graph = molToTree(myMol).first;
    cout << "graph" << endl;
    for (int i = 0; i < graph.size(); ++i) {
        cout << i << " : ";
        for (int j = 0; j < graph[i].size(); ++j) {
            cout << graph[i][j] << ' ';
        }
        cout << endl;
    }
    
    auto res = getPrufferCode(graph);
    cout << "prufferCode" << endl;
    for (int i = 0; i < res.size(); ++i) {
        cout << res[i] << ' ';
    }
    
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
