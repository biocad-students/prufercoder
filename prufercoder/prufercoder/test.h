//
//  test.h
//  prufercoder
//
//  Created by Юрий Кравченко on 28/07/16.
//  Copyright © 2016 Юрий Кравченко. All rights reserved.
//

#ifndef test_h
#define test_h


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

void testPrufferEncoding() {
    
    vector<int> result = getPrufferCode(graph);
    
    /*for (int i = 0; i < result.size(); ++i) {
     std::cout << result[i] << ' ';
     }*/
    assert(result == prufferCode);
}

void testPrufferDecoding() {
    
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
    assert(tree == graph);
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

void test() {
    testInit();
    testPrufferEncoding();
    testPrufferDecoding();
    atomCodeTest();
}

#endif /* test_h */
