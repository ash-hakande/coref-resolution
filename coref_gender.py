import pickle

gd = {}
def fillDict():
	global gd
	fp = open("genderDict", 'r')
	gd = pickle.load(fp)


def search(word):
	global gd
	t1 = ()
	t2 = ()
	t3 = ()
	if(word in gd): t1 = gd[word]
	if ('! '+word) in gd: t2 = gd['! '+ word]
	if (word+ ' !') in gd: t3 = gd[word+ ' !']

	return t1, t2, t3



def isUnmappable_gender(Mention1, Mention2, headWord1, headWord2):
	pass


def isUnmappable_number(Mention1, Mention2, headWord1, headWord2):
	pass


def main(Mention1, Mention2, headWord1, headWord2):
	if isUnmappable(Mention1, Mention2) == False:
		return True						#None
	else: return False


fillDict()
while(True):
	print search(raw_input())