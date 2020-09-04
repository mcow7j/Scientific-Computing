""" Your college id here:
    Template code for part 2, contains 3 functions:
    codonToAA: returns amino acid corresponding to input amino acid
    DNAtoAA: to be completed for part 2.1
    pairSearch: to be completed for part 2.2
"""


def codonToAA(codon):
	"""Return amino acid corresponding to input codon.
	Assumes valid codon has been provided as input
	"_" is returned for valid codons that do not
	correspond to amino acids.
	"""
	table = {
		'ATA':'i', 'ATC':'i', 'ATT':'i', 'ATG':'m',
		'ACA':'t', 'ACC':'t', 'ACG':'t', 'ACT':'t',
		'AAC':'n', 'AAT':'n', 'AAA':'k', 'AAG':'k',
		'AGC':'s', 'AGT':'s', 'AGA':'r', 'AGG':'r',
		'CTA':'l', 'CTC':'l', 'CTG':'l', 'CTT':'l',
		'CCA':'p', 'CCC':'p', 'CCG':'p', 'CCT':'p',
		'CAC':'h', 'CAT':'h', 'CAA':'q', 'CAG':'q',
		'CGA':'r', 'CGC':'r',+ 'CGG':'r', 'CGT':'r',
		'GTA':'v', 'GTC':'v', 'GTG':'v', 'GTT':'v',
		'GCA':'a', 'GCC':'a', 'GCG':'a', 'GCT':'a',
		'GAC':'d', 'GAT':'d', 'GAA':'e', 'GAG':'e',
		'GGA':'g', 'GGC':'g', 'GGG':'g', 'GGT':'g',
		'TCA':'s', 'TCC':'s', 'TCG':'s', 'TCT':'s',
		'TTC':'f', 'TTT':'f', 'TTA':'l', 'TTG':'l',
		'TAC':'y', 'TAT':'y', 'TAA':'_', 'TAG':'_',
		'TGC':'c', 'TGT':'c', 'TGA':'_', 'TGG':'w',
	}
	return table[codon]


def DNAtoAA(S):
    """Convert genetic sequence contained in input string, S,
    into string of amino acids corresponding to the distinct
    amino acids found in S and listed in the order that
    they appear in S
    """
        #set up a string for final list and
    w=set()
    fin_list=""
    n=len(S)
    #iterate through list
    for i in range(0,n,3):
        x=codonToAA(S[i:i+3])
        if not x in w:
            w.add(x)
            fin_list+=x
    return fin_list

def baseconv(S):
  """Convert gene test_sequence    string to list of ints    """
  c={}
  c['A']="0"
  c['C']="1"
  c['G']="2"
  c['T']="3"
  L=""
  for s in S:
    L+=c[s]
  return L

def sumhash(S,M):
  """converts list of ints into rolling hash score for first M elements"""
  hi=0
  for i in range(0,M):
    hi=hi+int(S[i])*(4**(M-i-1))
  return hi



def pairSearch(L,pairs):
    """Find locations within adjacent strings (contained in input list,L)
    that match k-mer pairs found in input list pairs. Each element of pairs
    is a 2-element tuple containing k-mer strings
    """
    #set up basic varablies
    N=len(L[0])
    M=len(pairs[0][0])
    locations=[]

    #convert lists into integrer 0,1,2,3
    for j in range(0,len(L)):
      L[j]=baseconv(L[j])

    #set up empty dictionary for pairs to be searched in list,
    #key is there hash value, item index in pairs
    listpairs1={k: [] for k in range(3*M*4**M+1)}
    listpairs2={k: [] for k in range(3*M*4**M+1)}
    for j in range(0,len(pairs)):
      listpairs1[sumhash(baseconv(pairs[j][0]),M)].append(j)
      listpairs2[sumhash(baseconv(pairs[j][1]),M)].append(j)

    #remove empty entries from dictionarys
    listpairs1 = {k:v for k,v in listpairs1.items() if v}
    listpairs2 =  {k:v for k,v in listpairs2.items() if v}

    #search first sequence recording where pairs come up in a dictionary
    S=L[0]
    listind={}
    hi=sumhash(S,M)
    bm=4**M
    for ind in range(1,N-M+1):
      hi=4*hi-(int(S[ind-1])*bm)+(int(S[ind-1+M]))
      if hi in listpairs1:
        listind[ind]=hi

    #search all remain sqeuences
    for j in range(1,len(L)):
        #store previous sequence index matrix
      listindold=listind
      #reset index matrix
      listind={}
      #calc first entry of rolling sum
      hi=sumhash(L[j],M)
      #rolling sum
      for ind in range(1,N-M+1):
        hi=4*hi-(int(L[j][ind-1])*bm)+(int(L[j][ind-1+M]))
        #check if its element of pairs1
        if hi in listpairs1:
          listind[ind]=hi
        #check if its element of pairs2
        if hi in listpairs2:
        #see if both pair agree on a common index w.r.t to hash value
          if ind in listindold:
            #calc intersection
            w=set(listpairs2[hi]) & set(listpairs1[listindold[ind]])
            #if w is non emtpty means a pair is aligned
            if w!={}:
              locations.append([ind,j-1,list(w)[0]])

    return locations
