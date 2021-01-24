#-*- coding:utf-8 -*-
import sys

try:
	sys.stdin.reconfigure(encoding='utf-8')
except:
	pass


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            if '▁▁' in line:
                line = line.strip().replace(' ', '').replace('▁▁', ' ').replace('▁', '').strip()
            else:
                line = line.strip().replace(' ', '').replace('▁', ' ').strip()

            sys.stdout.write(line + '\n')
        else:
            sys.stdout.write('\n')
