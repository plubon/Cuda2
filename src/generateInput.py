import random

def main():
	file = open("testfile", "w")
	for row in xrange(1000000):
		file.write(str(random.randrange(-1000000, 1000000)))
		file.write("\n")
	file.close()

if __name__ == "__main__":
    main()