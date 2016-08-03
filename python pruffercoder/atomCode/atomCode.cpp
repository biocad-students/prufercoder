//
//  atomCode.cpp
//  prufercoder
//
//  Created by Юрий Кравченко on 28/07/16.
//  Copyright © 2016 Юрий Кравченко. All rights reserved.
//

#include "atomCode.h"
#include <vector>
#include <string>

using namespace std;

const vector<string> atomSet = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Br", "Cl", "B", "C", "N", "O", "P", "S", "F", "I", "H", "+", "-"};

int atomEncoding(const string &s) {
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