#!/usr/bin/env python3
import sys
sys.path.insert(0, 'student')
from optimizer import ABC_FLOWS, AIG_FLOWS
print('ABC_FLOWS:', len(ABC_FLOWS))
for k in ABC_FLOWS:
    print(' ', k)
print()
print('AIG_FLOWS:', len(AIG_FLOWS))
for k in AIG_FLOWS:
    print(' ', k)
