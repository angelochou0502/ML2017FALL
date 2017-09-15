import sys

count = 0
countForWord = []
appearWord = []

file = open(sys.argv[1],'r')
text = file.read()
for word in text.split():
	if word in appearWord:
		countForWord[appearWord.index(word)] += 1
	else:
		appearWord.append(word)
		countForWord.append(1)

out_file = open('Q1.txt','w')
for i, word in enumerate(appearWord):
	out_file.write("%s %d %d"%(appearWord[i],i,countForWord[i]))
	if(i != len(appearWord) - 1):
		out_file.write("\n")
	out_file.flush()

file.close()
out_file.close()