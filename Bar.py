import sys
import time


def progressBar( bar_length=20):
    for i in range(101):
        percent = float(i) / 100
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


def pro():
    for i in range(51):
        s1 = "\r[%s%s]%d%%" % ("*" * (i), " " * (50 - i), i*2)
        sys.stdout.write(s1)
        sys.stdout.flush()
        time.sleep(0.02)

if __name__ == '__main__':
    pro()