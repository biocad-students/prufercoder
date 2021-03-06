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
#include "pruffer.h"
#include "atomCode.h"

using namespace std;

void createRings() {
    set<string> mols;
    vector<string> names = {"frequent.txt", "common.txt", "in-man.txt", "in-drugs.txt", "endogenous.txt", "in-nature.txt", "rare.txt", "single.txt", "double.txt", "triple.txt", "quadruple.txt", "quintuple.txt", "six-plus.txt", "for-sale.txt", "not-for-sale.txt"};
    /*vector<string> names = {"six-plus.txt"};
    for (int i = 1; i <= 31; ++i) {
        names.push_back("six-plus-" + to_string(i) + ".txt");
    }*/
    for (int i = 0; i < names.size(); ++i) {
        ifstream fin("../data/" + names[i]);
        string s;
        while (getline(fin, s)) {
            vector<string> strings;
            boost::split(strings, s, boost::is_any_of("\t"));
            mols.insert(strings[1]);
        }
    }
    //ofstream fout("data/all-six-plus.txt");
    ofstream fout("../data/allRings.txt");
    //ofstream fout("data/popularRings.txt");
    cerr << mols.size() << endl;
    for (auto i = mols.begin(); i != mols.end(); ++i) {
        fout << *i << endl;
    }
}

//reading rings.txt
vector<RDKit::ROMol*> getRings() {
    vector<RDKit::ROMol*> result;
    ifstream fin("../data/popularRings.txt");
    int i = 0;
    while (!fin.eof()) {
        string mol;
        fin >> mol;
        result.push_back(RDKit::SmilesToMol(mol));
        i++;
        if (i % 1000 == 0) {
            cerr << i << endl;
        }
    }
    return result;
}

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

//#include "test.h"

int main(int argc, const char * argv[]) {
    
    //test();
    
    RDKit::ROMol * myMol = RDKit::SmilesToMol("CC(C)C[C@H](NC(=O)[C@@H](CC1=CNC2=CC=CC=C12)NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@H](CO)NC(=O)[C@H](CC1=CNC2=C1C=CC=C2)NC(=O)[C@H](CC1=CNC=N1)NC(=O)[C@@H]1CCC(=O)N1)C(=O)N[C@@H](CCCNC(N)=N)C(=O)N1CCC[C@H]1C(=O)NCC(N)=O");
    
    auto data = molToTree(myMol);
    auto graph = data.first;
    
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
    cout << endl;
    
    cout << "mapping" << endl;
    for (int i = 0; i < data.second.size(); ++i) {
        cout << data.second[i] << ' ';
    }
    
    return 0;
}
